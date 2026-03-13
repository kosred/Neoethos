"""Compatibility shim for legacy genetic strategy interfaces."""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from . import fast_backtest as fb

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_TALIB_POP = hasattr(_fb, "evaluate_population_talib_ohlcv")
except Exception:
    _fb = None  # type: ignore
    _RUST_TALIB_POP = False


@dataclass(slots=True)
class GeneticGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]]
    weights: dict[str, float]
    fitness: float = 0.0
    evaluated: bool = False


class GeneticStrategyEvolution:
    """
    Legacy wrapper retained for compatibility with older strategy tests/integrations.
    """

    def __init__(self, population_size: int = 50, mixer: Any | None = None) -> None:
        self.population_size = max(1, int(population_size or 1))
        self.mixer = mixer
        self.population: list[GeneticGene] = []

    @staticmethod
    def _strict_rust_requested() -> bool:
        # Legacy wrapper is now Rust-only.
        return True

    @staticmethod
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
        if np.issubdtype(arr.dtype, np.datetime64):
            return True
        if arr.dtype.kind == "O":
            for item in arr.tolist():
                if item is None:
                    continue
                if hasattr(item, "year") and hasattr(item, "month") and hasattr(item, "day"):
                    return True
                try:
                    np.datetime64(item, "ns")
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
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
            if hasattr(idx, "view"):
                viewed = idx.view("int64")
                if hasattr(viewed, "to_numpy"):
                    return np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
                return np.asarray(viewed, dtype=np.int64).reshape(-1)
        except Exception:
            pass
        try:
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        except Exception:
            return np.zeros(arr.size, dtype=np.int64)

    @staticmethod
    def _rust_time_index_arrays(idx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
            return None
        ns = GeneticStrategyEvolution._index_to_ns_int64(idx)
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

    @staticmethod
    def _datetime_index_to_unix_ms(idx: Any) -> np.ndarray | None:
        if not GeneticStrategyEvolution._is_datetime_index(idx):
            return None
        rust = GeneticStrategyEvolution._rust_time_index_arrays(idx)
        if rust is not None:
            unix_ms, _month_idx, _day_idx = rust
            return unix_ms
        ns = GeneticStrategyEvolution._index_to_ns_int64(idx)
        if ns.size <= 0:
            return np.zeros(0, dtype=np.int64)
        return (np.asarray(ns, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)

    @staticmethod
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

    @staticmethod
    def _frame_len(frame: Any) -> int:
        try:
            return int(len(frame))
        except Exception:
            return 0

    @staticmethod
    def _frame_columns(frame: Any) -> list[str]:
        cols = getattr(frame, "columns", None)
        if cols is None:
            return []
        try:
            return [str(c) for c in list(cols)]
        except Exception:
            return []

    @staticmethod
    def _frame_attr(frame: Any, key: str, default: Any = None) -> Any:
        attrs = getattr(frame, "attrs", None)
        if isinstance(attrs, dict):
            return attrs.get(key, default)
        return default

    @classmethod
    def _resolve_column_name(cls, frame: Any, name: str) -> str | None:
        target = str(name).strip().lower()
        if not target:
            return None
        for col in cls._frame_columns(frame):
            if str(col).strip().lower() == target:
                return col
        return None

    @staticmethod
    def _to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
        if hasattr(values, "to_numpy"):
            with np.errstate(all="ignore"):
                arr = values.to_numpy(dtype=dtype, copy=False)  # type: ignore[call-arg]
        else:
            arr = np.asarray(values, dtype=dtype)
        return np.asarray(arr, dtype=dtype).reshape(-1)

    @classmethod
    def _fit_len(cls, values: Any, n: int, *, fill: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
        out = cls._to_numpy_1d(values, dtype=dtype)
        target = max(0, int(n))
        if out.size == target:
            return out
        if out.size <= 0:
            return np.full(target, float(fill), dtype=dtype)
        if out.size > target:
            return out[:target]
        pad = np.full(target - out.size, float(out[-1]), dtype=dtype)
        return np.concatenate([out, pad])

    @classmethod
    def _frame_column_numpy(
        cls,
        frame: Any,
        name: str,
        *,
        n_rows: int,
        dtype: Any = np.float64,
        default: Any | None = None,
    ) -> np.ndarray:
        col = cls._resolve_column_name(frame, name)
        if col is None:
            if default is None:
                raise KeyError(name)
            return cls._fit_len(default, n_rows, fill=0.0, dtype=dtype)
        values = frame[col]  # type: ignore[index]
        return cls._fit_len(values, n_rows, fill=0.0, dtype=dtype)

    @classmethod
    def _month_day_indices(cls, idx: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        rust = cls._rust_time_index_arrays(idx)
        if rust is not None:
            _unix_ms, month_idx, day_idx = rust
            return month_idx[:n], day_idx[:n]
        if cls._is_datetime_index(idx):
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
        ns = cls._index_to_ns_int64(idx)
        if ns.size > 0:
            vmax = int(np.max(np.abs(ns)))
            if vmax > 10**14:
                dt = np.asarray(ns, dtype=np.int64).astype("datetime64[ns]")
                month_idx = dt.astype("datetime64[M]").astype(np.int64)
                day_idx = dt.astype("datetime64[D]").astype(np.int64)
                return month_idx[:n], day_idx[:n]
        seq = np.arange(n, dtype=np.int64)
        return seq, seq

    def _evaluate_population(self, df: Any, population: list[GeneticGene] | None = None) -> None:
        genes = population if population is not None else self.population
        if not genes:
            return

        if self._frame_empty(df):
            for gene in genes:
                gene.fitness = float("-inf")
                gene.evaluated = True
            return

        n_rows = self._frame_len(df)
        close = self._frame_column_numpy(df, "close", n_rows=n_rows, dtype=np.float64)
        high = self._frame_column_numpy(df, "high", n_rows=n_rows, dtype=np.float64, default=close)
        low = self._frame_column_numpy(df, "low", n_rows=n_rows, dtype=np.float64, default=close)
        idx = getattr(df, "index", None)
        month_idx, day_idx = self._month_day_indices(idx, n_rows)

        symbol = str(self._frame_attr(df, "symbol", "") or "")
        pip_size, pip_val = fb.infer_pip_metrics(symbol)

        # Rust-first population path: indicators -> features -> signals -> eval in Rust.
        if _RUST_TALIB_POP and _fb is not None:
            try:
                open_ = self._frame_column_numpy(df, "open", n_rows=n_rows, dtype=np.float64, default=close)
                volume = None
                if self._resolve_column_name(df, "volume") is not None:
                    volume = self._frame_column_numpy(df, "volume", n_rows=n_rows, dtype=np.float64)
                timestamps = self._datetime_index_to_unix_ms(idx)
                indicator_sets: list[list[str]] = []
                weight_sets: list[list[float]] = []
                long_thr: list[float] = []
                short_thr: list[float] = []
                sl_arr: list[float] = []
                tp_arr: list[float] = []
                rust_genes: list[GeneticGene] = []

                for gene in genes:
                    inds = [str(i).upper() for i in (gene.indicators or []) if str(i).strip()]
                    if not inds:
                        gene.fitness = float("-inf")
                        gene.evaluated = True
                        continue
                    rust_genes.append(gene)
                    indicator_sets.append(inds)
                    weights_map = gene.weights or {}
                    weight_sets.append(
                        [float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0) for ind in inds]
                    )
                    long_thr.append(0.66)
                    short_thr.append(-0.66)
                    sl_arr.append(30.0)
                    tp_arr.append(60.0)

                if rust_genes:
                    metrics = _fb.evaluate_population_talib_ohlcv(  # type: ignore[attr-defined]
                        open_,
                        high,
                        low,
                        close,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thr,
                        short_thresholds=short_thr,
                        sl_pips=sl_arr,
                        tp_pips=tp_arr,
                        timestamps=timestamps,
                        volume=volume,
                        include_raw=True,
                        smc_gate_threshold=float(np.float32(0.0)),
                        max_hold_bars=0,
                        trailing_enabled=False,
                        trailing_atr_multiplier=1.0,
                        trailing_be_trigger_r=1.0,
                        pip_value=float(pip_size),
                        spread_pips=1.5,
                        commission_per_trade=7.0,
                        pip_value_per_lot=float(pip_val),
                        causal_min_bars=30,
                    )
                    metrics_arr = np.asarray(metrics, dtype=np.float64)
                    if (
                        metrics_arr.ndim == 2
                        and metrics_arr.shape[0] == len(rust_genes)
                        and metrics_arr.shape[1] >= 1
                    ):
                        for i, gene in enumerate(rust_genes):
                            fit = float(metrics_arr[i, 0]) if np.isfinite(metrics_arr[i, 0]) else float("-inf")
                            gene.fitness = fit
                            gene.evaluated = bool(np.isfinite(fit))
                        if all(g.evaluated for g in genes):
                            return
            except Exception:
                # Preserve compatibility by dropping to the legacy Python signal/eval path.
                pass

        # Rust bridge path: if full population evaluator is unavailable, still compute signals in Rust bulk
        # and run batch backtest in one vectorized call.
        if _fb is not None and hasattr(_fb, "talib_bulk_signals_ohlcv"):
            try:
                pending = [g for g in genes if not g.evaluated]
                if pending:
                    open_ = self._frame_column_numpy(df, "open", n_rows=n_rows, dtype=np.float64, default=close)
                    volume = None
                    if self._resolve_column_name(df, "volume") is not None:
                        volume = self._frame_column_numpy(df, "volume", n_rows=n_rows, dtype=np.float64)
                    timestamps = self._datetime_index_to_unix_ms(idx)

                    indicator_sets: list[list[str]] = []
                    weight_sets: list[list[float]] = []
                    long_thr: list[float] = []
                    short_thr: list[float] = []
                    mapped: list[GeneticGene] = []
                    for gene in pending:
                        inds = [str(i).upper() for i in (gene.indicators or []) if str(i).strip()]
                        if not inds:
                            gene.fitness = float("-inf")
                            gene.evaluated = True
                            continue
                        mapped.append(gene)
                        indicator_sets.append(inds)
                        weights_map = gene.weights or {}
                        weight_sets.append(
                            [float(weights_map.get(ind, weights_map.get(ind.lower(), 1.0)) or 1.0) for ind in inds]
                        )
                        long_thr.append(0.66)
                        short_thr.append(-0.66)

                    if mapped:
                        try:
                            causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
                        except Exception:
                            causal_min_bars = 30
                        causal_min_bars = max(2, causal_min_bars)
                        try:
                            raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                                open_,
                                high,
                                low,
                                close,
                                indicator_sets=indicator_sets,
                                weight_sets=weight_sets,
                                long_thresholds=long_thr,
                                short_thresholds=short_thr,
                                timestamps=timestamps,
                                volume=volume,
                                include_raw=False,
                                causal_min_bars=causal_min_bars,
                            )
                        except TypeError:
                            raw = _fb.talib_bulk_signals_ohlcv(  # type: ignore[attr-defined]
                                open_,
                                high,
                                low,
                                close,
                                indicator_sets=indicator_sets,
                                weight_sets=weight_sets,
                                long_thresholds=long_thr,
                                short_thresholds=short_thr,
                                timestamps=timestamps,
                                volume=volume,
                                include_raw=False,
                            )
                        sig = np.asarray(raw, dtype=np.int8)
                        n_bars = int(n_rows)
                        n_genes = int(len(mapped))
                        if sig.ndim == 2:
                            if sig.shape[0] == n_bars and sig.shape[1] == n_genes:
                                sig = sig.T
                            elif not (sig.shape[0] == n_genes and sig.shape[1] == n_bars):
                                sig = np.empty((0, 0), dtype=np.int8)
                        else:
                            sig = np.empty((0, 0), dtype=np.int8)

                        if sig.ndim == 2 and sig.shape[0] == n_genes and sig.shape[1] == n_bars:
                            close_mat = np.ascontiguousarray(np.broadcast_to(close, (n_genes, close.size)), dtype=np.float64)
                            high_mat = np.ascontiguousarray(np.broadcast_to(high, (n_genes, high.size)), dtype=np.float64)
                            low_mat = np.ascontiguousarray(np.broadcast_to(low, (n_genes, low.size)), dtype=np.float64)
                            month_mat = np.ascontiguousarray(np.broadcast_to(month_idx, (n_genes, month_idx.size)), dtype=np.int64)
                            day_mat = np.ascontiguousarray(np.broadcast_to(day_idx, (n_genes, day_idx.size)), dtype=np.int64)
                            sl_arr = np.full(n_genes, 30.0, dtype=np.float64)
                            tp_arr = np.full(n_genes, 60.0, dtype=np.float64)
                            metrics = fb.batch_evaluate_strategies(
                                close_mat,
                                high_mat,
                                low_mat,
                                sig,
                                month_mat,
                                day_mat,
                                sl_arr,
                                tp_arr,
                                pip_value=pip_size,
                                pip_value_per_lot=pip_val,
                                spread_pips=1.5,
                                commission_per_trade=7.0,
                            )
                            metrics_arr = np.asarray(metrics, dtype=np.float64)
                            if metrics_arr.ndim == 2 and metrics_arr.shape[0] == n_genes and metrics_arr.shape[1] >= 1:
                                for i, gene in enumerate(mapped):
                                    fit = float(metrics_arr[i, 0]) if np.isfinite(metrics_arr[i, 0]) else float("-inf")
                                    gene.fitness = fit
                                    gene.evaluated = bool(np.isfinite(fit))
                                if all(g.evaluated for g in genes):
                                    return
            except Exception:
                pass

        for gene in genes:
            if gene.evaluated:
                continue
            gene.fitness = float("-inf")
            gene.evaluated = True


__all__ = ["GeneticGene", "GeneticStrategyEvolution"]
