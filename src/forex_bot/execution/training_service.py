from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
import concurrent.futures
import gc
import json
import logging
import multiprocessing
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..core.config import ALL_TIMEFRAMES, Settings
from ..data.loader import DataLoader
from ..data.news.client import get_sentiment_analyzer
from ..domain.events import PreparedDataset
from ..features.pipeline import FeatureEngineer
from ..strategy.discovery import AutonomousDiscoveryEngine
from ..training.trainer import ModelTrainer

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore
except Exception:  # pragma: no cover - optional native extension
    _fb = None  # type: ignore


def _index_to_ns_int64(index: Any) -> np.ndarray | None:
    if index is None:
        return None
    try:
        if hasattr(index, "asi8"):
            arr = np.asarray(index.asi8, dtype=np.int64).reshape(-1)
            return arr if arr.size > 0 else None
    except Exception:
        pass
    try:
        arr = np.asarray(index).reshape(-1)
        if arr.size <= 0:
            return None
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        if hasattr(index, "view"):
            viewed = index.view("int64")
            if hasattr(viewed, "to_numpy"):
                v_arr = np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
            else:
                v_arr = np.asarray(viewed, dtype=np.int64).reshape(-1)
            return v_arr if v_arr.size > 0 else None
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
    return out if out.size > 0 else None


def _rust_time_index_arrays(index: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
        return None
    idx_ns = _index_to_ns_int64(index)
    if idx_ns is None or idx_ns.size <= 0:
        return None
    try:
        unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(np.asarray(idx_ns, dtype=np.int64))
    except Exception:
        return None
    return (
        np.asarray(unix_ms, dtype=np.int64).reshape(-1),
        np.asarray(month_idx, dtype=np.int64).reshape(-1),
        np.asarray(day_idx, dtype=np.int64).reshape(-1),
    )


def _rust_align_ffill_by_ns(
    src_idx: np.ndarray,
    src_vals: np.ndarray,
    tgt_idx: np.ndarray,
    *,
    fill: float,
) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_ffill_values_by_ns"):
        return None
    try:
        out = _fb.align_ffill_values_by_ns(
            np.asarray(src_idx, dtype=np.int64),
            np.asarray(src_vals, dtype=np.float64),
            np.asarray(tgt_idx, dtype=np.int64),
            float(fill),
        )
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.size != int(np.asarray(tgt_idx).size):
        return None
    return arr


def _rust_align_exact_by_ns(
    src_idx: np.ndarray,
    src_vals: np.ndarray,
    tgt_idx: np.ndarray,
    *,
    fill: float,
) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_exact_values_by_ns"):
        return None
    try:
        out = _fb.align_exact_values_by_ns(
            np.asarray(src_idx, dtype=np.int64),
            np.asarray(src_vals, dtype=np.float64),
            np.asarray(tgt_idx, dtype=np.int64),
            float(fill),
        )
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.size != int(np.asarray(tgt_idx).size):
        return None
    return arr


def _rust_align_feature_matrix(
    src_matrix: np.ndarray,
    src_col_idx: np.ndarray,
    dst_col_idx: np.ndarray,
    *,
    dst_width: int,
) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_feature_matrix"):
        return None
    src = np.asarray(src_matrix, dtype=np.float32)
    if src.ndim != 2:
        return None
    try:
        out = _fb.align_feature_matrix(
            src,
            np.asarray(src_col_idx, dtype=np.int64),
            np.asarray(dst_col_idx, dtype=np.int64),
            int(max(0, dst_width)),
        )
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float32)
    rows = int(src.shape[0])
    width = int(max(0, dst_width))
    if arr.ndim != 2 or arr.shape != (rows, width):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _rust_sorted_index_order(index: Any) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "sorted_index_order"):
        return None
    idx_ns = _index_to_ns_int64(index)
    if idx_ns is None:
        return None
    try:
        out = _fb.sorted_index_order(np.asarray(idx_ns, dtype=np.int64))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != idx_ns.size:
        return None
    return order


def _sorted_time_order(index_like: Any, n_rows: int) -> np.ndarray | None:
    idx_ns = _index_to_ns_int64(index_like)
    if idx_ns is None or idx_ns.size != int(n_rows) or idx_ns.size <= 1:
        return None
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None:
        return order
    return np.argsort(idx_ns, kind="mergesort")


def _rust_rank_scores_desc(scores: Any, *, absolute: bool = False) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "rank_scores_desc"):
        return None
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    try:
        out = _fb.rank_scores_desc(arr, bool(absolute))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != arr.size:
        return None
    return order


def _is_dataframe(value: Any) -> bool:
    return bool(
        hasattr(value, "columns")
        and hasattr(value, "index")
        and callable(getattr(value, "to_numpy", None))
    )


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))


def _is_series(value: Any) -> bool:
    return bool(hasattr(value, "index") and hasattr(value, "to_numpy") and not hasattr(value, "columns"))


def _is_datetime_index(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return True
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
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
    except Exception:
        return False


def _tabular_module(*, required: bool = True):
    _ = required
    return None


def _make_series(
    values: Any,
    *,
    index: Any | None = None,
    dtype: Any | None = None,
    template: Any | None = None,
) -> Any:
    if template is not None:
        ctor = getattr(template, "__class__", None)
        if ctor is not None:
            with contextlib.suppress(Exception):
                if dtype is None:
                    return ctor(values, index=index)
                return ctor(values, index=index, dtype=dtype)
    arr = np.asarray(values)
    if dtype is not None:
        with contextlib.suppress(Exception):
            arr = arr.astype(dtype, copy=False)
    return arr.reshape(-1)


def _make_dataframe(
    data: Any,
    *,
    columns: Any | None = None,
    index: Any | None = None,
    template: Any | None = None,
) -> Any:
    if template is not None:
        ctor = getattr(template, "__class__", None)
        if ctor is not None:
            kwargs: dict[str, Any] = {}
            if columns is not None:
                kwargs["columns"] = columns
            if index is not None:
                kwargs["index"] = index
            with contextlib.suppress(Exception):
                return ctor(data, **kwargs)
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    n_rows = int(arr.shape[0]) if arr.ndim > 0 else 0
    if columns is None:
        names = [f"f{i}" for i in range(int(arr.shape[1]) if arr.ndim == 2 else 0)]
    else:
        names = [str(c) for c in list(columns)]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out_data: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        if arr.ndim == 2 and j < arr.shape[1]:
            out_data[name] = np.asarray(arr[:, j]).reshape(-1)
        else:
            out_data[name] = np.zeros(n_rows, dtype=np.float32)
    idx_obj = np.asarray(index).reshape(-1) if index is not None else np.arange(n_rows, dtype=np.int64)
    if idx_obj.size != n_rows:
        idx_obj = np.arange(n_rows, dtype=np.int64)
    return _NumpyFrame(out_data, index=idx_obj)


def _range_index(n: int) -> Any:
    return np.arange(max(0, int(n)), dtype=np.int64)


def _to_datetime_index(values: Any) -> Any:
    arr = np.asarray(values).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype="datetime64[ns]")
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]", copy=False)
    if arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
        vmax = int(np.max(np.abs(vals))) if vals.size > 0 else 0
        if vmax > 10**14:
            return vals.astype("datetime64[ns]")
        if vmax > 10**11:
            return vals.astype("datetime64[ms]").astype("datetime64[ns]")
        return vals.astype("datetime64[s]").astype("datetime64[ns]")
    with np.errstate(all="ignore"):
        try:
            return arr.astype("datetime64[ns]")
        except Exception:
            return np.arange(arr.size, dtype=np.int64).astype("datetime64[s]").astype("datetime64[ns]")


def _concat_dataframes(items: list[Any]) -> Any | None:
    if not items:
        return None
    out = items[0]
    for item in items[1:]:
        append_fn = getattr(out, "_append", None)
        if callable(append_fn):
            with contextlib.suppress(Exception):
                out = append_fn(item)
                continue
        append_fn = getattr(out, "append", None)
        if callable(append_fn):
            with contextlib.suppress(Exception):
                out = append_fn(item)
                continue
        if not _is_frame_like(out):
            return None
        cols: list[str] = []
        seen: set[str] = set()
        for frame in items:
            for col in _frame_columns(frame):
                if col not in seen:
                    seen.add(col)
                    cols.append(col)
        if not cols:
            return None

        def _fit_len_any(values: Any, n: int) -> np.ndarray:
            arr = np.asarray(values).reshape(-1)
            target = max(0, int(n))
            if arr.size == target:
                return arr
            if arr.size <= 0:
                return np.full(target, np.nan, dtype=np.float64)
            if arr.size > target:
                return arr[:target]
            pad = np.full(target - arr.size, arr[-1], dtype=arr.dtype if arr.dtype != object else object)
            return np.concatenate([arr, pad])

        out_data: dict[str, np.ndarray] = {}
        idx_parts: list[np.ndarray] = []
        for frame in items:
            n = _frame_len(frame)
            idx_obj = _frame_index(frame)
            idx_arr = np.asarray(idx_obj).reshape(-1) if idx_obj is not None else np.arange(n, dtype=np.int64)
            idx_parts.append(_fit_len_any(idx_arr, n))
        for col in cols:
            chunks: list[np.ndarray] = []
            for frame in items:
                n = _frame_len(frame)
                src_col = _frame_resolve_column(frame, col)
                if src_col is None:
                    chunks.append(np.full(n, np.nan, dtype=np.float64))
                    continue
                with contextlib.suppress(Exception):
                    raw = frame[src_col]  # type: ignore[index]
                    chunks.append(_fit_len_any(raw, n))
                    continue
                chunks.append(np.full(n, np.nan, dtype=np.float64))
            out_data[col] = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)

        idx_all = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64)
        ctor = getattr(items[0], "__class__", None)
        if ctor is not None:
            with contextlib.suppress(Exception):
                return ctor(out_data, index=idx_all)
        return _NumpyFrame(out_data, index=idx_all)
    return out


def _compact_ohlcv_metadata_frame(
    meta: Any,
    *,
    symbol: str | None = None,
) -> Any | None:
    if meta is None or not (_is_dataframe(meta) or _is_frame_like(meta)):
        return None
    data: dict[str, np.ndarray] = {}
    for col in ("open", "high", "low", "close", "volume"):
        arr = _frame_column_numpy_optional(meta, col, dtype=np.float64)
        if arr is None:
            continue
        data[str(col)] = np.asarray(arr, dtype=np.float64).reshape(-1)
    if "close" not in data:
        return None
    attrs = getattr(meta, "attrs", None)
    out_attrs = dict(attrs) if isinstance(attrs, dict) else {}
    sym = str(symbol or out_attrs.get("symbol", "") or "").strip()
    if sym:
        out_attrs["symbol"] = sym
    return _NumpyFrame(
        data,
        index=_frame_index(meta),
        attrs=out_attrs or None,
    )


class _NumpyFrame:
    """Minimal frame-like object for strict frame-native discovery flows."""

    def __init__(
        self,
        data: dict[str, Any],
        *,
        index: Any | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        n = 0
        if self._data:
            try:
                n = int(next(iter(self._data.values())).shape[0])
            except Exception:
                n = 0
        if index is None:
            self.index = np.arange(n, dtype=np.int64)
        else:
            idx = np.asarray(index).reshape(-1)
            if idx.size == n:
                self.index = idx
            elif idx.size <= 0:
                self.index = np.arange(n, dtype=np.int64)
            elif idx.size > n:
                self.index = idx[:n]
            else:
                pad = np.full(n - idx.size, idx[-1], dtype=idx.dtype)
                self.index = np.concatenate([idx, pad])
        self.columns = list(self._data.keys())
        self.attrs: dict[str, Any] = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        name = str(key)
        vals = np.asarray(value).reshape(-1)
        n = int(len(self))
        if vals.size != n:
            vals = _fit_len_array(vals, n, fill=0.0, dtype=vals.dtype if vals.size > 0 else np.float32)
        self._data[name] = vals
        if name not in self.columns:
            self.columns.append(name)

    def copy(self, deep: bool = False) -> "_NumpyFrame":
        _ = deep
        out = _NumpyFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            index=np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )
        return out

    def to_numpy(self, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if not self.columns:
            out = np.zeros((len(self), 0), dtype=np.float32)
        else:
            mats = [np.asarray(self._data[c]).reshape(-1) for c in self.columns]
            out = np.column_stack(mats) if mats else np.zeros((len(self), 0), dtype=np.float32)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        if copy:
            out = np.asarray(out).copy()
        return np.asarray(out)

    def tail(self, n: int) -> "_NumpyFrame":
        take = max(0, int(n))
        if take <= 0:
            return _NumpyFrame(
                {k: v[:0] for k, v in self._data.items()},
                index=self.index[:0],
                attrs=dict(self.attrs),
            )
        return _NumpyFrame(
            {k: v[-take:] for k, v in self._data.items()},
            index=self.index[-take:],
            attrs=dict(self.attrs),
        )


def _fit_len_array(values: Any, n: int, *, fill: float = 0.0, dtype: Any = np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    target = max(0, int(n))
    if arr.size == target:
        return arr
    if arr.size <= 0:
        return np.full(target, float(fill), dtype=dtype)
    if arr.size > target:
        return arr[:target]
    pad = np.full(target - arr.size, float(arr[-1]), dtype=dtype)
    return np.concatenate([arr, pad])


def _frame_empty(obj: Any) -> bool:
    if obj is None:
        return True
    try:
        return bool(obj.empty)
    except Exception:
        pass
    try:
        return int(len(obj)) <= 0
    except Exception:
        return True


def _frame_len(obj: Any) -> int:
    try:
        return int(len(obj))
    except Exception:
        return 0


def _frame_copy(obj: Any) -> Any:
    if obj is None:
        return None
    with contextlib.suppress(Exception):
        return obj.copy(deep=False)
    with contextlib.suppress(Exception):
        return obj.copy()
    return obj


def _frame_columns(obj: Any) -> list[str]:
    cols = getattr(obj, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_has_column(obj: Any, name: str) -> bool:
    target = str(name).strip().lower()
    for col in _frame_columns(obj):
        if str(col).strip().lower() == target:
            return True
    return False


def _frame_resolve_column(obj: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(obj):
        if str(col).strip().lower() == target:
            return col
    return None


def _frame_index(obj: Any) -> Any | None:
    return getattr(obj, "index", None)


def _to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        with contextlib.suppress(Exception):
            out = values.to_numpy(dtype=dtype, copy=False)
            return np.asarray(out, dtype=dtype).reshape(-1)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _frame_column_numpy(obj: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    col = _frame_resolve_column(obj, name)
    if col is None:
        raise KeyError(name)
    return _to_numpy_1d(obj[col], dtype=dtype)


def _frame_column_numpy_optional(obj: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray | None:
    with contextlib.suppress(Exception):
        return _frame_column_numpy(obj, name, dtype=dtype)
    return None


def _frame_set_column(obj: Any, name: str, values: Any, *, dtype: Any = np.float32) -> bool:
    vals = np.asarray(values, dtype=dtype).reshape(-1)
    with contextlib.suppress(Exception):
        obj[str(name)] = vals
        return True
    data = getattr(obj, "_data", None)
    if isinstance(data, dict):
        key = str(name)
        n = _frame_len(obj)
        data[key] = _fit_len_array(vals, n, fill=0.0, dtype=dtype)
        cols = getattr(obj, "columns", None)
        if isinstance(cols, list) and key not in cols:
            cols.append(key)
        return True
    return False


def _frame_to_2d_float32(
    obj: Any,
    *,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if obj is None:
        return np.zeros((0, 0), dtype=np.float32), []

    if _is_dataframe(obj):
        names = [str(c) for c in list(obj.columns)]
        try:
            arr = obj.to_numpy(dtype=np.float32, copy=False)
        except Exception:
            arr = np.asarray(obj)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), names

    if _is_frame_like(obj):
        names = _frame_columns(obj)
        n_rows = _frame_len(obj)
        mats: list[np.ndarray] = []
        resolved: list[str] = []
        for name in names:
            with contextlib.suppress(Exception):
                vec = _to_numpy_1d(obj[name], dtype=np.float32)  # type: ignore[index]
                mats.append(_fit_len_array(vec, n_rows, fill=0.0, dtype=np.float32))
                resolved.append(str(name))
        if mats:
            arr = np.column_stack(mats).astype(np.float32, copy=False)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr, resolved
        return np.zeros((n_rows, 0), dtype=np.float32), []

    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    names = list(feature_names or [])
    if len(names) != int(arr.shape[1]):
        names = [f"f{i}" for i in range(int(arr.shape[1]))]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), names


def _slice_rows_positions(obj: Any, rows: np.ndarray) -> Any:
    if obj is None:
        return None
    take = np.asarray(rows, dtype=np.int64).reshape(-1)
    if _is_dataframe(obj):
        with contextlib.suppress(Exception):
            return obj.take(take)
        with contextlib.suppress(Exception):
            base_idx = np.asarray(getattr(obj, "index")).reshape(-1)
            return obj.loc[base_idx[take]]
    if _is_frame_like(obj):
        idx = _frame_index(obj)
        idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(_frame_len(obj), dtype=np.int64)
        out_data: dict[str, np.ndarray] = {}
        for col in _frame_columns(obj):
            with contextlib.suppress(Exception):
                vec = np.asarray(obj[col]).reshape(-1)  # type: ignore[index]
                out_data[str(col)] = vec[take]
        attrs = getattr(obj, "attrs", None)
        return _NumpyFrame(
            out_data,
            index=idx_arr[take] if idx_arr.size > 0 else np.asarray([], dtype=np.int64),
            attrs=dict(attrs) if isinstance(attrs, dict) else None,
        )
    arr = np.asarray(obj)
    return arr[take]


def _slice_rows_range(obj: Any, start: int, end: int) -> Any:
    if obj is None:
        return None
    s = max(0, int(start))
    e = max(s, int(end))
    if _is_dataframe(obj):
        take = np.arange(s, e, dtype=np.int64)
        with contextlib.suppress(Exception):
            return obj.take(take)
        with contextlib.suppress(Exception):
            base_idx = np.asarray(getattr(obj, "index")).reshape(-1)
            return obj.loc[base_idx[take]]
    if _is_frame_like(obj):
        return _slice_rows_positions(obj, np.arange(s, e, dtype=np.int64))
    arr = np.asarray(obj)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[s:e]


def _series_like_to_int8(
    values: Any,
    *,
    row_index: Any | None = None,
    n_rows: int | None = None,
) -> np.ndarray:
    src = values
    arr: np.ndarray | None = None
    if row_index is not None:
        src_idx = _index_to_ns_generic(getattr(src, "index", None))
        tgt_idx = _index_to_ns_generic(row_index)
        if src_idx is not None and tgt_idx is not None:
            src_vals = _to_numpy_1d(src, dtype=np.float32)
            aligned = _align_exact_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)
            if aligned is not None:
                arr = aligned
    if arr is None:
        arr = _to_numpy_1d(src, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int8, copy=False)
    if n_rows is not None:
        return _fit_len_array(arr, int(max(0, n_rows)), fill=0.0, dtype=np.int8)
    return arr


def _index_to_ns_generic(index: Any) -> np.ndarray | None:
    return _index_to_ns_int64(index)


def _align_ffill_by_ns(
    src_idx: np.ndarray | None,
    src_vals: np.ndarray,
    tgt_idx: np.ndarray | None,
    *,
    dtype: Any = np.float32,
    fill: float = 0.0,
) -> np.ndarray | None:
    if src_idx is None or tgt_idx is None:
        return None
    s_idx = np.asarray(src_idx, dtype=np.int64).reshape(-1)
    t_idx = np.asarray(tgt_idx, dtype=np.int64).reshape(-1)
    vals = np.asarray(src_vals, dtype=np.float64).reshape(-1)
    if s_idx.size <= 0 or t_idx.size <= 0 or vals.size <= 0:
        return np.full(t_idx.size, float(fill), dtype=dtype)
    m = min(s_idx.size, vals.size)
    s_idx = s_idx[:m]
    vals = vals[:m]
    rust_out = _rust_align_ffill_by_ns(s_idx, vals, t_idx, fill=float(fill))
    if rust_out is not None:
        return np.nan_to_num(
            rust_out,
            nan=float(fill),
            posinf=float(fill),
            neginf=float(fill),
        ).astype(dtype, copy=False)
    order = _sorted_time_order(s_idx, s_idx.size)
    if order is not None:
        s_idx = s_idx[order]
        vals = vals[order]
    pos = np.searchsorted(s_idx, t_idx, side="right") - 1
    out = np.full(t_idx.size, float(fill), dtype=np.float64)
    valid = pos >= 0
    if np.any(valid):
        out[valid] = vals[np.clip(pos[valid], 0, vals.size - 1)]
    return np.nan_to_num(out, nan=float(fill), posinf=float(fill), neginf=float(fill)).astype(dtype, copy=False)


def _align_exact_by_ns(
    src_idx: np.ndarray | None,
    src_vals: np.ndarray,
    tgt_idx: np.ndarray | None,
    *,
    dtype: Any = np.float32,
    fill: float = 0.0,
) -> np.ndarray | None:
    if src_idx is None or tgt_idx is None:
        return None
    s_idx = np.asarray(src_idx, dtype=np.int64).reshape(-1)
    t_idx = np.asarray(tgt_idx, dtype=np.int64).reshape(-1)
    vals = np.asarray(src_vals, dtype=np.float64).reshape(-1)
    out = np.full(t_idx.size, float(fill), dtype=np.float64)
    if s_idx.size <= 0 or t_idx.size <= 0 or vals.size <= 0:
        return out.astype(dtype, copy=False)
    m = min(s_idx.size, vals.size)
    s_idx = s_idx[:m]
    vals = vals[:m]
    rust_out = _rust_align_exact_by_ns(s_idx, vals, t_idx, fill=float(fill))
    if rust_out is not None:
        return np.nan_to_num(
            rust_out,
            nan=float(fill),
            posinf=float(fill),
            neginf=float(fill),
        ).astype(dtype, copy=False)
    order = _sorted_time_order(s_idx, s_idx.size)
    if order is not None:
        s_idx = s_idx[order]
        vals = vals[order]
    pos = np.searchsorted(s_idx, t_idx, side="left")
    valid = pos < s_idx.size
    if np.any(valid):
        matched = np.zeros(t_idx.size, dtype=bool)
        vp = pos[valid]
        matched[valid] = s_idx[vp] == t_idx[valid]
        take = valid & matched
        if np.any(take):
            out[take] = vals[pos[take]]
    return np.nan_to_num(out, nan=float(fill), posinf=float(fill), neginf=float(fill)).astype(dtype, copy=False)


def _column_index_mapping(
    src_names: list[str],
    dst_name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for src_i, raw_name in enumerate(src_names):
        dst_i = dst_name_to_idx.get(str(raw_name))
        if dst_i is None:
            continue
        src_cols.append(int(src_i))
        dst_cols.append(int(dst_i))
    if not src_cols or not dst_cols:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.asarray(src_cols, dtype=np.int64), np.asarray(dst_cols, dtype=np.int64)


def _align_feature_matrix(
    src_matrix: Any,
    src_col_idx: np.ndarray,
    dst_col_idx: np.ndarray,
    *,
    dst_width: int,
) -> np.ndarray:
    src = np.asarray(src_matrix, dtype=np.float32)
    if src.ndim == 0:
        src = src.reshape(1, 1)
    elif src.ndim == 1:
        src = src.reshape(-1, 1)
    elif src.ndim != 2:
        src = src.reshape(src.shape[0], -1)
    rows = int(src.shape[0])
    width = int(max(0, dst_width))
    if rows <= 0 or width <= 0:
        return np.zeros((rows, width), dtype=np.float32)

    src_idx = np.asarray(src_col_idx, dtype=np.int64).reshape(-1)
    dst_idx = np.asarray(dst_col_idx, dtype=np.int64).reshape(-1)
    m = min(int(src_idx.size), int(dst_idx.size))
    if m <= 0:
        return np.zeros((rows, width), dtype=np.float32)
    src_idx = src_idx[:m]
    dst_idx = dst_idx[:m]

    rust = _rust_align_feature_matrix(
        src,
        src_idx,
        dst_idx,
        dst_width=width,
    )
    if rust is not None:
        return rust

    out = np.zeros((rows, width), dtype=np.float32)
    out[:, dst_idx] = src[:, src_idx]
    return out


def _rust_sort_dedup_rows_by_index(
    x: np.ndarray,
    y: np.ndarray,
    idx_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "sort_dedup_rows_by_index"):
        return None
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8).reshape(-1)
    idx_arr = np.asarray(idx_ns, dtype=np.int64).reshape(-1)
    if x_arr.ndim != 2:
        return None
    rows = int(min(x_arr.shape[0], y_arr.size, idx_arr.size))
    if rows <= 0:
        return (
            np.zeros((0, x_arr.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    try:
        out_x, out_y, out_idx = _fb.sort_dedup_rows_by_index(
            x_arr[:rows],
            y_arr[:rows],
            idx_arr[:rows],
        )
    except Exception:
        return None
    x_out = np.asarray(out_x, dtype=np.float32)
    y_out = np.asarray(out_y, dtype=np.int8).reshape(-1)
    idx_out = np.asarray(out_idx, dtype=np.int64).reshape(-1)
    if x_out.ndim != 2:
        return None
    if x_out.shape[0] != y_out.size or y_out.size != idx_out.size:
        return None
    return (
        np.nan_to_num(x_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        y_out.astype(np.int8, copy=False),
        idx_out.astype(np.int64, copy=False),
    )


def _sort_dedup_rows_by_index(
    x: np.ndarray,
    y: np.ndarray,
    idx_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8).reshape(-1)
    idx_arr = np.asarray(idx_ns, dtype=np.int64).reshape(-1)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    rows = int(min(x_arr.shape[0], y_arr.size, idx_arr.size))
    if rows <= 0:
        return (
            np.zeros((0, x_arr.shape[1] if x_arr.ndim == 2 else 0), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    x_arr = x_arr[:rows]
    y_arr = y_arr[:rows]
    idx_arr = idx_arr[:rows]

    rust = _rust_sort_dedup_rows_by_index(x_arr, y_arr, idx_arr)
    if rust is not None:
        return rust

    order = _sorted_time_order(idx_arr, idx_arr.size)
    if order is not None:
        x_arr = x_arr[order]
        y_arr = y_arr[order]
        idx_arr = idx_arr[order]
    x_sorted = x_arr
    y_sorted = y_arr
    idx_sorted = idx_arr
    keep = np.ones(idx_sorted.shape[0], dtype=bool)
    if idx_sorted.shape[0] > 1:
        keep[1:] = idx_sorted[1:] != idx_sorted[:-1]
    return x_sorted[keep], y_sorted[keep], idx_sorted[keep]


class TrainingService:
    """
    Manages model training, feature engineering for training,
    and strategy discovery cycles.
    """

    def __init__(
        self,
        settings: Settings,
        data_loader: DataLoader,
        trainer: ModelTrainer,
        feature_engineer: FeatureEngineer,
        discovery_engine: AutonomousDiscoveryEngine,
        autotune_hints: Any,
    ):
        self.settings = settings
        self.data_loader = data_loader
        self.trainer = trainer
        self.feature_engineer = feature_engineer
        self.discovery_engine = discovery_engine
        self.autotune_hints = autotune_hints
        self._ray_started = False
        self._progress_path = self.trainer.models_dir / "global_incremental_progress.json"
        self._prop_search_task: asyncio.Task | None = None
        self._discovery_task: asyncio.Task | None = None
        self._prop_search_thread: threading.Thread | None = None
        self._discovery_thread: threading.Thread | None = None

    def _start_background_thread(self, name: str, target) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _start_prop_search_thread(self, symbols: list[str]) -> None:
        if self._prop_search_thread and self._prop_search_thread.is_alive():
            logger.info("[STRATEGY DISCOVERY] Async discovery already running; skipping new launch.")
            return

        def _runner() -> None:
            try:
                asyncio.run(self._run_prop_search_for_symbols(symbols, stop_event=None))
            except Exception as exc:
                logger.warning(f"[STRATEGY DISCOVERY] Background prop search failed: {exc}", exc_info=True)

        logger.info("[STRATEGY DISCOVERY] Running prop search in background thread.")
        self._prop_search_thread = self._start_background_thread("forex-prop-search", _runner)

    def _load_progress(self) -> set[str]:
        """Load completed-symbol list to allow resuming long incremental runs."""
        try:
            data = json.loads(Path(self._progress_path).read_text())
            if isinstance(data, list):
                return {str(s) for s in data}
        except FileNotFoundError:
            return set()
        except Exception as exc:
            logger.warning(f"Failed to load incremental progress: {exc}")
        return set()

    def _save_progress(self, completed: set[str]) -> None:
        try:
            Path(self._progress_path).write_text(json.dumps(sorted(completed)))
        except Exception as exc:
            logger.warning(f"Failed to persist incremental progress: {exc}")

    def _get_prop_max_rows(self, tf: str | None = None) -> int:
        try:
            max_rows = int(getattr(self.settings.models, "prop_search_max_rows", 0) or 0)
        except Exception:
            max_rows = 0
        try:
            by_tf = getattr(self.settings.models, "prop_search_max_rows_by_tf", {}) or {}
        except Exception:
            by_tf = {}
        if tf and isinstance(by_tf, dict):
            tf_key = str(tf).upper()
            for key, value in by_tf.items():
                if str(key).upper() == tf_key:
                    try:
                        return int(value or 0)
                    except Exception:
                        return max_rows
        return max_rows

    def _prop_search_async_enabled(self) -> bool:
        raw = os.environ.get("FOREX_BOT_PROP_SEARCH_ASYNC", "")
        if str(raw).strip() != "":
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
        try:
            return bool(getattr(self.settings.models, "prop_search_async", False))
        except Exception:
            return False

    def _prop_search_async_wait(self) -> bool:
        raw = os.environ.get("FOREX_BOT_PROP_SEARCH_ASYNC_WAIT")
        if raw is None or str(raw).strip() == "":
            try:
                return bool(getattr(self.settings.models, "prop_search_async_wait", False))
            except Exception:
                return False
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _discovery_async_enabled(self) -> bool:
        raw = os.environ.get("FOREX_BOT_DISCOVERY_ASYNC")
        if raw is None or str(raw).strip() == "":
            # Default to async when prop search is async (keeps event loop alive).
            return self._prop_search_async_enabled()
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _quick_e2e_enabled(self) -> bool:
        raw = os.environ.get("FOREX_BOT_QUICK_E2E", "")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _rust_only_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        if profile.startswith("rust"):
            return True
        tree_backend = str(os.environ.get("FOREX_BOT_TREE_BACKEND", "") or "").strip().lower()
        if tree_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
            return True
        features_backend = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "") or "").strip().lower()
        return features_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}

    def _normalize_discovery_budget(
        self,
        *,
        experts: int,
        iterations: int,
        has_gpu: bool,
    ) -> tuple[int, int]:
        quick = self._quick_e2e_enabled()
        if has_gpu:
            experts = int(experts)
            iterations = int(iterations)
        else:
            # Keep discovery responsive on CPU-only nodes by default.
            experts = min(int(experts), 40)
            iterations = min(int(iterations), 250)

        if quick:
            # Fast E2E: preserve discovery path but shrink search budget heavily.
            experts = min(int(experts), 4)
            iterations = min(int(iterations), 20)
            experts = max(2, int(experts))
            iterations = max(5, int(iterations))
        else:
            experts = max(8, int(experts))
            iterations = max(50, int(iterations))
        return experts, iterations

    def _prop_search_workers_override(self) -> int | None:
        for key in ("FOREX_BOT_DISCOVERY_CPU_BUDGET", "FOREX_BOT_PROP_SEARCH_WORKERS"):
            raw = os.environ.get(key)
            if not raw:
                continue
            try:
                value = int(raw)
            except Exception:
                continue
            if value > 0:
                return value
        return None

    async def _run_prop_search_for_symbols(
        self,
        symbols: list[str],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if not symbols:
            return
        if not bool(getattr(self.settings.models, "prop_search_enabled", False)):
            return
        try:
            from ..strategy.evo_prop import run_evo_search

            def _parse_tfs_env(name: str) -> list[str] | None:
                raw = os.environ.get(name)
                if not raw:
                    return None
                parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
                return parts or None

            def _parse_syms_env(name: str) -> list[str] | None:
                raw = os.environ.get(name)
                if not raw:
                    return None
                parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
                return parts or None

            tfs_env = _parse_tfs_env("FOREX_BOT_PROP_SEARCH_TFS")
            syms_env = _parse_syms_env("FOREX_BOT_PROP_SEARCH_SYMBOLS")

            symbols_to_run = symbols
            if syms_env:
                symbols_to_run = [s for s in syms_env if s in symbols]
            if not symbols_to_run:
                logger.warning("[STRATEGY DISCOVERY] No symbols available, skipping")
                return

            workers_override = self._prop_search_workers_override()
            if workers_override:
                logger.info(
                    "[STRATEGY DISCOVERY] Using %s CPU workers for prop search.",
                    workers_override,
                )

            logger.info(f"[STRATEGY DISCOVERY] Symbols: {symbols_to_run}")

            for sym in symbols_to_run:
                if stop_event and stop_event.is_set():
                    break
                logger.info(f"[STRATEGY DISCOVERY] Running prop-aware search on {sym} data...")

                self.settings.system.symbol = sym
                await self.data_loader.ensure_history(sym)
                frames = await self.data_loader.get_training_data(sym)
                if not isinstance(frames, dict) or not frames:
                    logger.warning(f"[STRATEGY DISCOVERY] No frames for {sym}, skipping")
                    continue

                base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")
                cfg_tfs = list(getattr(self.settings.system, "required_timeframes", []) or [])
                if not cfg_tfs:
                    cfg_tfs = list(getattr(self.settings.system, "higher_timeframes", []) or [])
                tfs = [base_tf]
                for tf in cfg_tfs:
                    if tf != base_tf:
                        tfs.append(tf)
                for tf in frames.keys():
                    if tf not in tfs:
                        tfs.append(tf)
                if tfs_env:
                    tfs = [tf for tf in tfs_env if tf in frames]

                if not tfs:
                    logger.warning(f"[STRATEGY DISCOVERY] {sym}: No timeframes available, skipping")
                    continue
                logger.info(f"[STRATEGY DISCOVERY] {sym}: Timeframes: {tfs}")

                for tf in tfs:
                    if stop_event and stop_event.is_set():
                        break
                    prop_df = frames.get(tf)
                    if prop_df is None or prop_df.empty:
                        continue

                    try:
                        prop_df.attrs["timeframe"] = tf
                        prop_df.attrs["tf"] = tf
                        prop_df.attrs["symbol"] = sym
                    except Exception:
                        pass

                    max_rows = self._get_prop_max_rows(tf)
                    if max_rows > 0 and len(prop_df) > max_rows:
                        prop_df = prop_df.tail(max_rows)
                        logger.info(
                            f"[STRATEGY DISCOVERY] {sym} {tf}: Using {len(prop_df):,} rows (capped)"
                        )

                    try:
                        pop = int(getattr(self.settings.models, "prop_search_population", 64) or 64)
                    except Exception:
                        pop = 64
                    try:
                        gens = int(getattr(self.settings.models, "prop_search_generations", 50) or 50)
                    except Exception:
                        gens = 50
                    try:
                        max_hours = float(getattr(self.settings.models, "prop_search_max_hours", 1.0) or 1.0)
                    except Exception:
                        max_hours = 1.0
                    try:
                        actual_balance = float(
                            getattr(self.settings.risk, "initial_balance", 100000.0) or 100000.0
                        )
                    except Exception:
                        actual_balance = 100000.0

                    checkpoint = str(
                        getattr(
                            self.settings.models,
                            "prop_search_checkpoint",
                            "models/strategy_evo_checkpoint.json",
                        )
                        or "models/strategy_evo_checkpoint.json"
                    )
                    ckpt_path = Path(checkpoint)
                    sym_tag = "".join(c for c in sym if c.isalnum() or c in ("-", "_"))
                    tf_tag = "".join(c for c in tf if c.isalnum() or c in ("-", "_"))
                    if len(symbols_to_run) > 1 or len(tfs) > 1:
                        checkpoint = str(
                            ckpt_path.with_name(f"{ckpt_path.stem}_{sym_tag}_{tf_tag}{ckpt_path.suffix}")
                        )

                    logger.info(
                        f"[STRATEGY DISCOVERY] {sym} {tf}: Config pop={pop}, gen={gens}, max_hours={max_hours:.1f}h, rows={len(prop_df):,}"
                    )

                    prop_settings = self.settings.model_copy()
                    prop_device = str(
                        getattr(self.settings.models, "prop_search_device", "cpu") or "cpu"
                    )
                    prop_settings.system.device = prop_device
                    prop_settings.system.symbol = sym

                    await asyncio.to_thread(
                        run_evo_search,
                        prop_df,
                        prop_settings,
                        pop,
                        gens,
                        checkpoint,
                        max_hours,
                        actual_balance,
                        max_workers=workers_override,
                    )
            logger.info("[STRATEGY DISCOVERY] Completed!")
        except Exception as exc:
            logger.warning(f"[STRATEGY DISCOVERY] Failed: {exc}", exc_info=True)

    @staticmethod
    def _safe_symbol_tag(symbol: str) -> str:
        safe = "".join(c for c in str(symbol or "") if c.isalnum() or c in ("-", "_"))
        return safe or "GLOBAL"

    def _prop_gene_artifact_paths(self, symbol: str) -> list[Path]:
        safe = self._safe_symbol_tag(symbol)
        paths: list[Path] = []
        cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache") or "cache")
        paths.append(cache_dir / f"talib_knowledge_{safe}.json")
        paths.append(cache_dir / "talib_knowledge.json")

        checkpoint = str(
            getattr(
                self.settings.models,
                "prop_search_checkpoint",
                "models/strategy_evo_checkpoint.json",
            )
            or "models/strategy_evo_checkpoint.json"
        )
        ckpt = Path(checkpoint)
        paths.append(ckpt)
        with contextlib.suppress(Exception):
            for candidate in ckpt.parent.glob(f"{ckpt.stem}_{safe}_*{ckpt.suffix}"):
                paths.append(candidate)

        uniq: list[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)

        existing = [p for p in uniq if p.exists() and p.is_file()]
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing

    def _load_prop_best_genes(self, symbol: str, max_genes: int = 100):
        try:
            from ..features.talib_mixer import TALibStrategyGene
        except Exception as exc:
            logger.debug("Prop gene load unavailable: %s", exc)
            return []
        rust_talib_available = False
        with contextlib.suppress(Exception):
            import forex_bindings  # type: ignore

            rust_talib_available = bool(hasattr(forex_bindings, "talib_bulk_signals_ohlcv"))
        if not rust_talib_available:
            logger.warning(
                "[STRATEGY DISCOVERY] %s: Rust TALib backend unavailable while loading discovered genes; skipping.",
                symbol,
            )
            return []

        try:
            max_genes = int(
                os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_GENES", str(max_genes)) or max_genes
            )
        except Exception:
            max_genes = int(max_genes)
        max_genes = max(1, max_genes)
        try:
            min_genes = int(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "100") or 100)
        except Exception:
            min_genes = 100
        min_genes = max(0, min_genes)
        if min_genes > max_genes:
            min_genes = max_genes
        strict_symbol = str(os.environ.get("FOREX_BOT_PROP_SYMBOL_STRICT", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        diversify_tf = str(os.environ.get("FOREX_BOT_PROP_DIVERSIFY_TIMEFRAMES", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            min_per_tf = int(os.environ.get("FOREX_BOT_PROP_MIN_PER_TF", "4") or 4)
        except Exception:
            min_per_tf = 4
        min_per_tf = max(0, min_per_tf)
        runtime_profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        elite_default = runtime_profile.startswith("rust")
        elite_filter = str(
            os.environ.get("FOREX_BOT_PROP_ELITE_FILTER", "1" if elite_default else "0") or ("1" if elite_default else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        strict_filtered_only = str(
            os.environ.get("FOREX_BOT_PROP_STRICT_FILTER", "1" if elite_filter else "0") or ("1" if elite_filter else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        require_forward = str(
            os.environ.get("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1" if elite_filter else "0") or ("1" if elite_filter else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        require_all_tfs = str(
            os.environ.get("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "1" if elite_filter else "0") or ("1" if elite_filter else "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        strict_tf_coverage = str(os.environ.get("FOREX_BOT_PROP_REQUIRE_ALL_TFS_STRICT", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            min_holdout_months = float(os.environ.get("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6.0" if elite_filter else "0.0") or (6.0 if elite_filter else 0.0))
        except Exception:
            min_holdout_months = 6.0 if elite_filter else 0.0
        min_holdout_months = float(max(0.0, min_holdout_months))
        try:
            holdout_max_dd = float(os.environ.get("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03" if elite_filter else "1.0") or (0.03 if elite_filter else 1.0))
        except Exception:
            holdout_max_dd = 0.03 if elite_filter else 1.0
        holdout_max_dd = float(min(1.0, max(0.0, holdout_max_dd)))
        try:
            min_truth = float(
                os.environ.get(
                    "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
                    os.environ.get(
                        "FOREX_BOT_MIN_TRUTH_PROBABILITY",
                        str(getattr(self.settings.models, "prop_search_holdout_min_truth_probability", 0.0) or 0.0),
                    ),
                )
                or 0.0
            )
        except Exception:
            min_truth = 0.0
        if min_truth > 1.0:
            min_truth *= 0.01
        min_truth = float(min(1.0, max(0.0, min_truth)))

        base_tf_cfg = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1").upper()
        expected_tfs: list[str] = [base_tf_cfg]
        if str(os.environ.get("FOREX_BOT_USE_ALL_TIMEFRAMES", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            for tf in list(getattr(self.settings.system, "multi_resolution_timeframes", []) or []):
                tfu = str(tf or "").upper()
                if tfu and tfu not in expected_tfs:
                    expected_tfs.append(tfu)
            for tf in ALL_TIMEFRAMES:
                tfu = str(tf or "").upper()
                if tfu and tfu not in expected_tfs:
                    expected_tfs.append(tfu)
        else:
            for tf in list(getattr(self.settings.system, "required_timeframes", []) or []):
                tfu = str(tf or "").upper()
                if tfu and tfu not in expected_tfs:
                    expected_tfs.append(tfu)
            for tf in list(getattr(self.settings.system, "higher_timeframes", []) or []):
                tfu = str(tf or "").upper()
                if tfu and tfu not in expected_tfs:
                    expected_tfs.append(tfu)

        if require_all_tfs and expected_tfs and max_genes < len(expected_tfs):
            logger.warning(
                "[STRATEGY DISCOVERY] %s: increasing max_genes from %s to %s to cover required timeframes.",
                symbol,
                max_genes,
                len(expected_tfs),
            )
            max_genes = len(expected_tfs)
            if min_genes > max_genes:
                min_genes = max_genes
        target_symbol = str(symbol or "").upper().strip()

        candidates = self._prop_gene_artifact_paths(symbol)
        if not candidates:
            return []

        available: set[str] | None = None

        def _to_float(source: dict[str, Any], key: str, default: float) -> float:
            try:
                return float(source.get(key, default) or default)
            except Exception:
                return float(default)

        def _to_bool(source: dict[str, Any], key: str, default: bool = False) -> bool:
            val = source.get(key, default)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return float(val) != 0.0
            return str(val).strip().lower() in {"1", "true", "yes", "on"}

        try:
            keep_max_dd = float(
                os.environ.get(
                    "FOREX_BOT_PROP_KEEP_MAX_DD",
                    getattr(self.settings.risk, "total_drawdown_limit", 0.07),
                )
                or 0.07
            )
        except Exception:
            keep_max_dd = 0.07
        keep_max_dd = float(min(1.0, max(0.0, keep_max_dd)))

        try:
            keep_min_profit = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_PROFIT", "0.0") or 0.0)
        except Exception:
            keep_min_profit = 0.0

        try:
            keep_min_trades = float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_TRADES", "1.0") or 1.0)
        except Exception:
            keep_min_trades = 1.0
        keep_min_trades = float(max(0.0, keep_min_trades))

        def _passes(gene: TALibStrategyGene) -> bool:
            profit = float(getattr(gene, "fitness", 0.0) or 0.0)
            if profit <= keep_min_profit:
                return False
            dd = float(getattr(gene, "max_dd_pct", 0.0) or 0.0)
            if dd > keep_max_dd:
                return False
            trades = float(getattr(gene, "trades", 0.0) or 0.0)
            if trades < keep_min_trades:
                return False
            if require_forward and not bool(getattr(gene, "forward_test_passed", False)):
                return False
            truth = float(getattr(gene, "truth_probability", 0.0) or 0.0)
            if truth > 1.0:
                truth *= 0.01
            if truth < min_truth:
                return False
            hold_months = float(getattr(gene, "holdout_months", 0.0) or 0.0)
            if min_holdout_months > 0.0 and hold_months < min_holdout_months:
                return False
            hold_dd = float(getattr(gene, "holdout_max_dd_pct", 0.0) or 0.0)
            if holdout_max_dd < 1.0 and hold_dd > holdout_max_dd:
                return False
            return True

        def _gene_key(gene: TALibStrategyGene) -> str:
            sid = str(getattr(gene, "strategy_id", "") or "").strip()
            if sid:
                return f"id:{sid}"
            return (
                f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
                f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
            )

        def _gene_tf(gene: TALibStrategyGene) -> str:
            tf = str(getattr(gene, "source_timeframe", "") or "").strip().upper()
            return tf or "UNK"

        def _top_up(base: list[TALibStrategyGene]) -> list[TALibStrategyGene]:
            if len(base) >= min_genes:
                return base[:max_genes]
            seen = {_gene_key(g) for g in base}
            out = list(base)
            for gene in merged:
                key = _gene_key(gene)
                if key in seen:
                    continue
                out.append(gene)
                seen.add(key)
                if len(out) >= min_genes or len(out) >= max_genes:
                    break
            return out[:max_genes]

        def _diversify_by_timeframe(base: list[TALibStrategyGene]) -> list[TALibStrategyGene]:
            if not diversify_tf or not base:
                return base[:max_genes]

            bucketed: dict[str, list[TALibStrategyGene]] = defaultdict(list)
            for gene in base:
                bucketed[_gene_tf(gene)].append(gene)
            if len(bucketed) <= 1:
                return base[:max_genes]

            # Timeframe order: strongest first, then round-robin for diversity.
            tf_order = sorted(
                bucketed.keys(),
                key=lambda tf: float(getattr(bucketed[tf][0], "fitness", 0.0) or 0.0),
                reverse=True,
            )
            cursors = {tf: 0 for tf in tf_order}
            out: list[TALibStrategyGene] = []

            # Pass 1: guarantee a minimum contribution per timeframe when possible.
            if min_per_tf > 0:
                for tf in tf_order:
                    bucket = bucketed[tf]
                    take = min(min_per_tf, len(bucket))
                    for i in range(take):
                        out.append(bucket[i])
                    cursors[tf] = take
                    if len(out) >= max_genes:
                        return out[:max_genes]

            # Pass 2: round-robin fill remainder.
            while len(out) < max_genes:
                advanced = False
                for tf in tf_order:
                    cur = cursors[tf]
                    bucket = bucketed[tf]
                    if cur >= len(bucket):
                        continue
                    out.append(bucket[cur])
                    cursors[tf] = cur + 1
                    advanced = True
                    if len(out) >= max_genes:
                        break
                if not advanced:
                    break
            return out[:max_genes]

        def _enforce_timeframe_coverage(
            base: list[TALibStrategyGene],
            *,
            fallback_pool: list[TALibStrategyGene],
        ) -> tuple[list[TALibStrategyGene], list[str]]:
            if not require_all_tfs:
                return base[:max_genes], []
            pool = fallback_pool if fallback_pool else base
            available = {_gene_tf(g) for g in pool if _gene_tf(g) != "UNK"}
            if not available:
                return base[:max_genes], []

            target = [tf for tf in expected_tfs if tf in available]
            if not target:
                target = sorted(available)
            cap = max(max_genes, len(target))

            out = list(base[:cap])
            seen = {_gene_key(g) for g in out}
            present = {_gene_tf(g) for g in out}

            for tf in target:
                if tf in present:
                    continue
                for gene in pool:
                    if _gene_tf(gene) != tf:
                        continue
                    key = _gene_key(gene)
                    if key in seen:
                        continue
                    out.append(gene)
                    seen.add(key)
                    present.add(tf)
                    break

            missing = [tf for tf in target if tf not in present]
            if missing and strict_tf_coverage:
                return [], missing
            return out[:cap], missing

        all_parsed: list[TALibStrategyGene] = []
        parsed_by_file: list[tuple[Path, int]] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload_symbol = str(payload.get("symbol", "") or "").upper().strip()
            if strict_symbol and payload_symbol and target_symbol and payload_symbol != target_symbol:
                continue
            payload_tf = str(payload.get("timeframe", payload.get("tf", "")) or "").upper().strip()
            raw_genes = payload.get("best_genes")
            if not isinstance(raw_genes, list) or not raw_genes:
                continue

            parsed: list[TALibStrategyGene] = []
            for raw in raw_genes:
                if not isinstance(raw, dict):
                    continue
                inds_raw = raw.get("indicators") or []
                indicators = []
                for ind in inds_raw:
                    name = str(ind).strip().upper()
                    if name and (not available or name in available) and name not in indicators:
                        indicators.append(name)
                if not indicators:
                    continue

                params_raw = raw.get("params") if isinstance(raw.get("params"), dict) else {}
                params = {}
                for ind in indicators:
                    val = params_raw.get(ind) or params_raw.get(ind.lower()) or params_raw.get(ind.upper()) or {}
                    params[ind] = dict(val) if isinstance(val, dict) else {}

                weights_raw = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
                weights = {}
                for ind in indicators:
                    w = weights_raw.get(ind)
                    if w is None:
                        w = weights_raw.get(ind.lower(), 1.0)
                    try:
                        weights[ind] = float(w)
                    except Exception:
                        weights[ind] = 1.0

                gene = TALibStrategyGene(
                    indicators=indicators,
                    params=params,
                    combination_method=str(raw.get("combination_method", "weighted_vote") or "weighted_vote"),
                    long_threshold=_to_float(raw, "long_threshold", 0.66),
                    short_threshold=_to_float(raw, "short_threshold", -0.66),
                    weights=weights,
                    preferred_regime=str(raw.get("preferred_regime", "any") or "any"),
                    strategy_id=str(raw.get("strategy_id", "") or ""),
                    fitness=_to_float(raw, "fitness", 0.0),
                    sharpe_ratio=_to_float(raw, "sharpe_ratio", 0.0),
                    win_rate=_to_float(raw, "win_rate", 0.0),
                    max_dd_pct=_to_float(
                        raw,
                        "max_dd_pct",
                        _to_float(raw, "max_drawdown", _to_float(raw, "max_dd", _to_float(raw, "drawdown", 0.0))),
                    ),
                    trades=_to_float(raw, "trades", _to_float(raw, "trades_count", _to_float(raw, "trade_count", 0.0))),
                    net_profit=_to_float(raw, "net_profit", 0.0),
                    profit_factor=_to_float(raw, "profit_factor", 0.0),
                    expectancy=_to_float(raw, "expectancy", 0.0),
                    use_ob=bool(raw.get("use_ob", False)),
                    use_fvg=bool(raw.get("use_fvg", False)),
                    use_liq_sweep=bool(raw.get("use_liq_sweep", False)),
                    mtf_confirmation=bool(raw.get("mtf_confirmation", False)),
                    use_premium_discount=bool(raw.get("use_premium_discount", False)),
                    use_inducement=bool(raw.get("use_inducement", False)),
                    tp_pips=_to_float(raw, "tp_pips", 40.0),
                    sl_pips=_to_float(raw, "sl_pips", 20.0),
                    source_symbol=payload_symbol,
                    source_timeframe=payload_tf,
                    in_sample_net_profit=_to_float(raw, "in_sample_net_profit", 0.0),
                    in_sample_sharpe_ratio=_to_float(raw, "in_sample_sharpe_ratio", 0.0),
                    in_sample_win_rate=_to_float(raw, "in_sample_win_rate", 0.0),
                    in_sample_profit_factor=_to_float(raw, "in_sample_profit_factor", 0.0),
                    in_sample_trades=_to_float(raw, "in_sample_trades", 0.0),
                    in_sample_max_dd_pct=_to_float(raw, "in_sample_max_dd_pct", 0.0),
                    in_sample_months=_to_float(raw, "in_sample_months", 0.0),
                    holdout_net_profit=_to_float(raw, "holdout_net_profit", 0.0),
                    holdout_sharpe_ratio=_to_float(raw, "holdout_sharpe_ratio", 0.0),
                    holdout_win_rate=_to_float(raw, "holdout_win_rate", 0.0),
                    holdout_profit_factor=_to_float(raw, "holdout_profit_factor", 0.0),
                    holdout_trades=_to_float(raw, "holdout_trades", 0.0),
                    holdout_max_dd_pct=_to_float(
                        raw,
                        "holdout_max_dd_pct",
                        _to_float(raw, "holdout_max_drawdown", _to_float(raw, "holdout_max_dd", 0.0)),
                    ),
                    holdout_months=_to_float(raw, "holdout_months", 0.0),
                    holdout_trades_per_month=_to_float(raw, "holdout_trades_per_month", 0.0),
                    holdout_monthly_profit_pct=_to_float(raw, "holdout_monthly_profit_pct", 0.0),
                    truth_probability=_to_float(raw, "truth_probability", 0.0),
                    forward_test_passed=_to_bool(
                        raw,
                        "forward_test_passed",
                        _to_bool(raw, "holdout_passed", False),
                    ),
                    in_sample_journal=dict(raw.get("in_sample_journal", {}) or {})
                    if isinstance(raw.get("in_sample_journal"), dict)
                    else {},
                    holdout_journal=dict(raw.get("holdout_journal", {}) or {})
                    if isinstance(raw.get("holdout_journal"), dict)
                    else {},
                )
                parsed.append(gene)

            if not parsed:
                continue
            parsed_by_file.append((path, len(parsed)))
            all_parsed.extend(parsed)

        if not all_parsed:
            return []

        dedup: dict[str, TALibStrategyGene] = {}
        for gene in sorted(all_parsed, key=lambda g: float(getattr(g, "fitness", 0.0) or 0.0), reverse=True):
            sid = str(getattr(gene, "strategy_id", "") or "").strip()
            if sid:
                key = f"id:{sid}"
            else:
                key = (
                    f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
                    f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
                )
            if key in dedup:
                continue
            dedup[key] = gene

        merged = list(dedup.values())
        merged.sort(
            key=lambda g: (
                float(getattr(g, "fitness", 0.0) or 0.0),
                float(getattr(g, "sharpe_ratio", 0.0) or 0.0),
                float(getattr(g, "win_rate", 0.0) or 0.0),
            ),
            reverse=True,
        )
        filtered = [g for g in merged if _passes(g)]

        if filtered:
            if strict_filtered_only:
                chosen = list(filtered[:max_genes])
            else:
                chosen = _top_up(filtered[:max_genes])
            if len(chosen) < min_genes:
                logger.warning(
                    "[STRATEGY DISCOVERY] %s: requested min_genes=%s but only %s strategies available after merge.",
                    symbol,
                    min_genes,
                    len(chosen),
                )
        else:
            if strict_filtered_only:
                logger.warning(
                    "[STRATEGY DISCOVERY] %s: strict strategy filter kept none; returning no strategies.",
                    symbol,
                )
                return []
            profitable = [g for g in merged if float(getattr(g, "fitness", 0.0) or 0.0) > 0.0]
            chosen = (profitable if profitable else merged)[:max_genes]
            chosen = _top_up(chosen)
            logger.warning(
                "[STRATEGY DISCOVERY] %s: strict strategy filter kept none; fallback to top-%s by fitness (min_genes=%s).",
                symbol,
                len(chosen),
                min_genes,
            )

        chosen = _diversify_by_timeframe(chosen)
        chosen, missing_tfs = _enforce_timeframe_coverage(chosen, fallback_pool=(filtered if filtered else merged))
        if missing_tfs:
            logger.warning(
                "[STRATEGY DISCOVERY] %s: timeframe coverage incomplete (missing=%s, strict=%s).",
                symbol,
                ",".join(missing_tfs),
                strict_tf_coverage,
            )
            if strict_tf_coverage:
                return []
        chosen_tfs = {str(getattr(g, "source_timeframe", "") or "UNK") for g in chosen}

        sources = ", ".join(f"{p.name}:{n}" for p, n in parsed_by_file[:6])
        if len(parsed_by_file) > 6:
            sources += ", ..."
        logger.info(
            "[STRATEGY DISCOVERY] %s: loaded %s genes from %s files (merged=%s, filtered=%s, selected=%s). "
            "Filters: profit>%.3f, max_dd<=%.3f, trades>=%.0f, min_genes=%s, strict_symbol=%s, diversify_tf=%s, "
            "require_forward=%s, strict_filter=%s, min_truth=%.2f, min_holdout_months=%.2f, holdout_max_dd=%.3f, require_all_tfs=%s, tf_count=%s. Sources: %s",
            symbol,
            sum(n for _, n in parsed_by_file),
            len(parsed_by_file),
            len(merged),
            len(filtered),
            len(chosen),
            keep_min_profit,
            keep_max_dd,
            keep_min_trades,
            min_genes,
            strict_symbol,
            diversify_tf,
            require_forward,
            strict_filtered_only,
            min_truth,
            min_holdout_months,
            holdout_max_dd,
            require_all_tfs,
            len(chosen_tfs),
            sources or "none",
        )
        return chosen

    def _apply_prop_discovered_base_signal(
        self,
        dataset: PreparedDataset,
        *,
        symbol: str,
        source_df: Any | None = None,
    ) -> PreparedDataset:
        if dataset is None:
            return dataset
        x_raw = getattr(dataset, "X", None)
        if x_raw is None:
            return dataset

        is_numpy_dataset = isinstance(x_raw, np.ndarray)
        if is_numpy_dataset:
            if x_raw.ndim != 2 or x_raw.shape[0] <= 0:
                return dataset
        else:
            if not (_is_dataframe(x_raw) or _is_frame_like(x_raw)):
                return dataset
            if _frame_empty(x_raw):
                return dataset

        try:
            max_genes = int(getattr(self.settings.models, "prop_search_portfolio_size", 4) or 4)
        except Exception:
            max_genes = 4
        max_genes = max(1, max_genes)
        genes = self._load_prop_best_genes(symbol=symbol, max_genes=max_genes)
        if not genes:
            return dataset

        source = None
        for candidate in (source_df, dataset.metadata, x_raw):
            if _frame_empty(candidate):
                continue
            cols = {str(c).lower() for c in _frame_columns(candidate)}
            if {"open", "high", "low", "close"}.issubset(cols):
                source = _frame_copy(candidate)
                if source is None:
                    source = candidate
                break
        if source is None:
            logger.warning(
                "[STRATEGY DISCOVERY] %s: cannot apply discovered base signal (missing OHLC source).",
                symbol,
            )
            return dataset

        src_idx = _frame_index(source)
        if _is_datetime_index(src_idx):
            order = _sorted_time_order(src_idx, len(source))
            if order is not None:
                source = _slice_rows_positions(source, order)

        try:
            threshold = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_THRESHOLD", "0.15") or 0.15)
        except Exception:
            threshold = 0.15
        threshold = float(min(0.95, max(0.0, threshold)))
        try:
            min_coverage = float(os.environ.get("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.005") or 0.005)
        except Exception:
            min_coverage = 0.005
        min_coverage = float(min(0.9, max(0.0, min_coverage)))

        row_count = int(x_raw.shape[0]) if is_numpy_dataset else int(len(x_raw))

        def _fit_len(values: np.ndarray, n: int, *, fill: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
            arr = np.asarray(values, dtype=dtype).reshape(-1)
            target = max(0, int(n))
            if arr.size == target:
                return arr
            if arr.size <= 0:
                return np.full(target, float(fill), dtype=dtype)
            if arr.size > target:
                return arr[:target]
            pad = np.full(target - arr.size, float(fill), dtype=dtype)
            return np.concatenate([arr, pad])

        def _apply_precomputed_signal(
            signal_values: np.ndarray | None,
            *,
            score_values: np.ndarray | None = None,
            backend: str = "rust",
        ) -> PreparedDataset | None:
            if signal_values is None:
                return None
            raw_sig = np.asarray(signal_values, dtype=np.int8).reshape(-1)
            if raw_sig.size <= 0:
                return None

            if is_numpy_dataset:
                signal = _fit_len(raw_sig, row_count, fill=0.0, dtype=np.int8)
                coverage = float(np.count_nonzero(signal)) / float(max(1, signal.size))
                if coverage < min_coverage:
                    logger.warning(
                        "[STRATEGY DISCOVERY] %s: discovered base signal too sparse (coverage=%.3f < %.3f), keeping existing base_signal.",
                        symbol,
                        coverage,
                        min_coverage,
                    )
                    return None

                x_np = np.asarray(x_raw, dtype=np.float32)
                feature_names = list(getattr(dataset, "feature_names", []) or [])
                if "base_signal" in feature_names:
                    base_idx = feature_names.index("base_signal")
                    x_np = x_np.copy()
                    x_np[:, base_idx] = signal.astype(np.float32, copy=False)
                else:
                    x_np = np.column_stack([x_np, signal.astype(np.float32, copy=False)]).astype(np.float32, copy=False)
                    feature_names.append("base_signal")

                logger.info(
                    "[STRATEGY DISCOVERY] %s: applied discovered base signal on numpy dataset via %s backend (%s genes, coverage=%.1f%%).",
                    symbol,
                    backend,
                    len(genes),
                    coverage * 100.0,
                )
                return PreparedDataset(
                    X=x_np,
                    y=dataset.y,
                    index=dataset.index,
                    feature_names=feature_names,
                    metadata=dataset.metadata,
                    labels=dataset.labels if dataset.labels is not None else dataset.y,
                )

            src_n = int(len(source.index))
            sig_src = _fit_len(raw_sig, src_n, fill=0.0, dtype=np.int8)
            src_idx_ns = self._index_to_ns(source.index)
            tgt_idx_ns = self._index_to_ns(x_raw.index)
            if src_idx_ns is not None and tgt_idx_ns is not None and src_idx_ns.size == sig_src.size:
                sig_aligned_f = self._align_values_ffill_by_timestamp(
                    src_idx_ns,
                    sig_src.astype(np.float32, copy=False),
                    tgt_idx_ns,
                    default=0.0,
                    dtype=np.float32,
                )
                sig_aligned = np.asarray(np.rint(sig_aligned_f), dtype=np.int8)
            else:
                sig_aligned = _fit_len(sig_src, int(len(x_raw.index)), fill=0.0, dtype=np.int8)
            coverage = float(np.count_nonzero(sig_aligned)) / float(max(1, sig_aligned.size))
            if coverage < min_coverage:
                logger.warning(
                    "[STRATEGY DISCOVERY] %s: discovered base signal too sparse (coverage=%.3f < %.3f), keeping existing base_signal.",
                    symbol,
                    coverage,
                    min_coverage,
                )
                return None

            if score_values is None:
                score_aligned = sig_aligned.astype(np.float32, copy=False)
            else:
                raw_score = np.asarray(score_values, dtype=np.float64).reshape(-1)
                score_src = _fit_len(raw_score, src_n, fill=0.0, dtype=np.float64)
                if src_idx_ns is not None and tgt_idx_ns is not None and src_idx_ns.size == score_src.size:
                    score_aligned = self._align_values_ffill_by_timestamp(
                        src_idx_ns,
                        score_src.astype(np.float32, copy=False),
                        tgt_idx_ns,
                        default=0.0,
                        dtype=np.float32,
                    )
                else:
                    score_aligned = _fit_len(score_src, int(len(x_raw.index)), fill=0.0, dtype=np.float32)
                score_aligned = np.asarray(
                    np.nan_to_num(score_aligned, nan=0.0, posinf=0.0, neginf=0.0),
                    dtype=np.float32,
                )

            x = x_raw.copy(deep=False)
            if "base_signal" in x.columns and "base_signal_static" not in x.columns:
                with contextlib.suppress(Exception):
                    raw = np.asarray(x["base_signal"])
                    if raw.dtype.kind in {"i", "u", "f", "b"}:
                        vals = raw.astype(np.float32, copy=False)
                    else:
                        flat = raw.reshape(-1)
                        tmp = np.empty(flat.shape[0], dtype=np.float32)
                        for i, v in enumerate(flat):
                            try:
                                tmp[i] = float(v)
                            except Exception:
                                tmp[i] = np.nan
                        vals = tmp.reshape(raw.shape)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    x["base_signal_static"] = vals.astype(np.int8, copy=False)
            x["base_signal"] = sig_aligned
            x["prop_signal_score"] = score_aligned

            logger.info(
                "[STRATEGY DISCOVERY] %s: applied discovered base signal via %s backend (%s genes, coverage=%.1f%%).",
                symbol,
                backend,
                len(genes),
                coverage * 100.0,
            )

            return PreparedDataset(
                X=x,
                y=dataset.y,
                index=getattr(x, "index", dataset.index),
                feature_names=list(x.columns),
                metadata=dataset.metadata,
                labels=dataset.labels if dataset.labels is not None else dataset.y,
            )

        rust_signal: np.ndarray | None = None
        try:
            rust_signal = self.feature_engineer._compute_discovered_base_signal_ohlcv_numpy(
                open_arr=np.asarray(source["open"], dtype=np.float64),
                high_arr=np.asarray(source["high"], dtype=np.float64),
                low_arr=np.asarray(source["low"], dtype=np.float64),
                close_arr=np.asarray(source["close"], dtype=np.float64),
                volume_arr=np.asarray(source["volume"], dtype=np.float64) if "volume" in source.columns else None,
                symbol=symbol,
            )
        except Exception as exc:
            logger.debug("Prop discovered signal (rust) failed for %s: %s", symbol, exc)
            rust_signal = None
        applied_rust = _apply_precomputed_signal(rust_signal, backend="rust")
        if applied_rust is not None:
            return applied_rust

        # Rust bulk TALib path: compute all gene signals in one bindings call and blend by gene fitness.
        rust_bulk_signal: np.ndarray | None = None
        rust_bulk_score: np.ndarray | None = None
        try:
            import forex_bindings as _fb  # type: ignore

            if hasattr(_fb, "talib_bulk_signals_ohlcv"):
                indicator_sets: list[list[str]] = []
                weight_sets: list[list[float]] = []
                long_thresholds: list[float] = []
                short_thresholds: list[float] = []
                fit_weights: list[float] = []
                for gene in genes:
                    inds = [str(i).upper() for i in (getattr(gene, "indicators", None) or []) if str(i).strip()]
                    if not inds:
                        continue
                    indicator_sets.append(inds)
                    g_weights = getattr(gene, "weights", {}) or {}
                    weight_sets.append(
                        [
                            float(g_weights.get(ind, g_weights.get(ind.lower(), 1.0)) or 1.0)
                            for ind in inds
                        ]
                    )
                    long_thresholds.append(float(getattr(gene, "long_threshold", 0.66)))
                    short_thresholds.append(float(getattr(gene, "short_threshold", -0.66)))
                    w = float(getattr(gene, "fitness", 0.0) or 0.0)
                    if not np.isfinite(w) or w <= 0.0:
                        w = 1.0
                    fit_weights.append(w)

                if indicator_sets:
                    timestamps = None
                    idx = source.index
                    if _is_datetime_index(idx):
                        if hasattr(idx, "view"):
                            idx_i64 = idx.view("int64")
                        elif hasattr(idx, "asi8"):
                            idx_i64 = idx.asi8
                        else:
                            idx_i64 = np.asarray(idx, dtype=np.int64)
                        if hasattr(idx_i64, "to_numpy"):
                            idx_i64 = idx_i64.to_numpy(dtype=np.int64, copy=False)
                        else:
                            idx_i64 = np.asarray(idx_i64, dtype=np.int64)
                        timestamps = (np.asarray(idx_i64, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)

                    open_arr = np.asarray(source["open"], dtype=np.float64)
                    high_arr = np.asarray(source["high"], dtype=np.float64)
                    low_arr = np.asarray(source["low"], dtype=np.float64)
                    close_arr = np.asarray(source["close"], dtype=np.float64)
                    volume_arr = (
                        np.asarray(source["volume"], dtype=np.float64)
                        if bool(getattr(self.settings.system, "use_volume_features", False)) and "volume" in source.columns
                        else None
                    )
                    try:
                        causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
                    except Exception:
                        causal_min_bars = 30
                    causal_min_bars = max(2, causal_min_bars)

                    try:
                        raw = _fb.talib_bulk_signals_ohlcv(
                            open_arr,
                            high_arr,
                            low_arr,
                            close_arr,
                            indicator_sets=indicator_sets,
                            weight_sets=weight_sets,
                            long_thresholds=long_thresholds,
                            short_thresholds=short_thresholds,
                            timestamps=timestamps,
                            volume=volume_arr,
                            include_raw=False,
                            causal_min_bars=causal_min_bars,
                        )
                    except TypeError:
                        raw = _fb.talib_bulk_signals_ohlcv(
                            open_arr,
                            high_arr,
                            low_arr,
                            close_arr,
                            indicator_sets=indicator_sets,
                            weight_sets=weight_sets,
                            long_thresholds=long_thresholds,
                            short_thresholds=short_thresholds,
                            timestamps=timestamps,
                            volume=volume_arr,
                            include_raw=False,
                        )

                    sig_mat = np.asarray(raw, dtype=np.float64)
                    n_src = int(len(source))
                    n_genes = int(len(indicator_sets))
                    if sig_mat.ndim == 2:
                        if sig_mat.shape[0] == n_src and sig_mat.shape[1] == n_genes:
                            pass
                        elif sig_mat.shape[0] == n_genes and sig_mat.shape[1] == n_src:
                            sig_mat = sig_mat.T
                        else:
                            sig_mat = np.empty((0, 0), dtype=np.float64)

                    if sig_mat.ndim == 2 and sig_mat.shape[0] == n_src and sig_mat.shape[1] == n_genes:
                        w = np.asarray(fit_weights, dtype=np.float64).reshape(-1)
                        w_sum = float(np.sum(np.abs(w)))
                        if w_sum > 0.0:
                            rust_bulk_score = (sig_mat @ w) / w_sum
                            rust_bulk_signal = np.where(
                                rust_bulk_score >= threshold,
                                1,
                                np.where(rust_bulk_score <= -threshold, -1, 0),
                            ).astype(np.int8, copy=False)
        except Exception as exc:
            logger.debug("Prop discovered signal (rust bulk) failed for %s: %s", symbol, exc)
            rust_bulk_signal = None
            rust_bulk_score = None

        applied_rust_bulk = _apply_precomputed_signal(
            rust_bulk_signal,
            score_values=rust_bulk_score,
            backend="rust_bulk_talib",
        )
        if applied_rust_bulk is not None:
            return applied_rust_bulk
        return dataset

    @staticmethod
    def _dataset_row_count(dataset: PreparedDataset | None) -> int:
        if dataset is None:
            return 0
        x = getattr(dataset, "X", None)
        if x is None:
            return 0
        try:
            return int(len(x))
        except Exception:
            return 0

    @staticmethod
    def _discovery_rust_features_enabled() -> bool:
        raw = os.environ.get("FOREX_BOT_DISCOVERY_RUST_FEATURES")
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
        profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        if profile.startswith("rust"):
            return True
        mode = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "") or "").strip().lower()
        if mode in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
            return True
        return str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _merge_ohlc_columns(
        target: Any,
        source: Any | None,
    ) -> Any:
        if _frame_empty(target) or _frame_empty(source):
            return target

        cols = [c for c in ("open", "high", "low", "close") if _frame_has_column(source, c)]
        if not cols:
            return target
        out = _frame_copy(target)
        if out is None:
            return target
        tgt_n = _frame_len(out)
        src_idx = _index_to_ns_generic(_frame_index(source))
        tgt_idx = _index_to_ns_generic(_frame_index(out))
        for col in cols:
            try:
                src_vals = _frame_column_numpy(source, col, dtype=np.float64)
            except Exception:
                continue
            aligned = _align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float64)
            if aligned is None:
                aligned = _fit_len_array(src_vals, tgt_n, fill=0.0, dtype=np.float64)
            _frame_set_column(out, col, aligned, dtype=np.float64)
        return out

    @staticmethod
    def _coerce_dataset_index(
        values: Any,
        rows: int,
        *,
        fallback_index: Any | None = None,
    ) -> Any:
        n = max(0, int(rows))
        if fallback_index is not None:
            with contextlib.suppress(Exception):
                if len(fallback_index) == n:
                    return fallback_index
        if values is None:
            return _range_index(n)
        if _is_datetime_index(values):
            with contextlib.suppress(Exception):
                idx = values
                if len(idx) > n:
                    return idx[:n]
                if len(idx) == n:
                    return idx
        arr = np.asarray(values).reshape(-1)
        if arr.size <= 0:
            return _range_index(n)
        if arr.size < n:
            if fallback_index is not None:
                with contextlib.suppress(Exception):
                    if len(fallback_index) == n:
                        return fallback_index
            return _range_index(n)
        arr = arr[:n]
        with contextlib.suppress(Exception):
            if np.issubdtype(arr.dtype, np.datetime64):
                return _to_datetime_index(arr)
        with contextlib.suppress(Exception):
            if arr.dtype.kind in {"i", "u"}:
                vmax = int(np.max(np.abs(arr.astype(np.int64, copy=False)))) if arr.size > 0 else 0
                if vmax > 10**14:
                    return _to_datetime_index(arr.astype(np.int64, copy=False))
                if vmax > 10**11:
                    return _to_datetime_index(arr.astype(np.int64, copy=False))
        with contextlib.suppress(Exception):
            return _to_datetime_index(arr)
        return _range_index(n)

    def _prepared_dataset_to_frame(
        self,
        dataset: PreparedDataset | None,
        *,
        fallback_frame: Any | None = None,
    ) -> Any | None:
        if dataset is None:
            return None
        x = getattr(dataset, "X", None)
        if x is None:
            return None
        if _is_dataframe(x):
            out = x.copy(deep=False)
        else:
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] <= 0:
                return None
            cols = list(getattr(dataset, "feature_names", []) or [])
            if len(cols) != int(arr.shape[1]):
                cols = [f"f{i}" for i in range(int(arr.shape[1]))]
            fb_idx = _frame_index(fallback_frame) if fallback_frame is not None else None
            idx = self._coerce_dataset_index(getattr(dataset, "index", None), int(arr.shape[0]), fallback_index=fb_idx)
            if fallback_frame is not None:
                out = _frame_copy(fallback_frame)
                if out is None:
                    out = fallback_frame
                tgt_n = _frame_len(out)
                src_idx = self._index_to_ns(getattr(dataset, "index", None))
                tgt_idx = self._index_to_ns(_frame_index(out))
                for i, col in enumerate(cols):
                    src_vals = np.asarray(arr[:, i], dtype=np.float32).reshape(-1)
                    aligned = _align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32)
                    if aligned is None:
                        aligned = _fit_len_array(src_vals, tgt_n, fill=0.0, dtype=np.float32)
                    _frame_set_column(out, col, aligned, dtype=np.float32)
            else:
                data = {str(col): np.asarray(arr[:, i], dtype=np.float32) for i, col in enumerate(cols)}
                out = _NumpyFrame(data, index=idx)

        if fallback_frame is not None:
            out = self._merge_ohlc_columns(out, fallback_frame)
        return out

    def _build_discovery_frames_for_tensor(
        self,
        frames: dict[str, Any],
        news_feats: Any | None,
        symbol: str | None,
        base_dataset: PreparedDataset | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build discovery frames with either base-TF propagation or full per-TF features."""
        full_tf = str(os.environ.get("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        base_tf = self.settings.system.base_timeframe

        cfg_tfs = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        if not cfg_tfs:
            cfg_tfs = list(getattr(self.settings.system, "required_timeframes", []) or [])
        timeframes = [base_tf] + cfg_tfs
        timeframes = [tf for tf in dict.fromkeys(timeframes) if tf in frames]
        for tf in frames.keys():
            if tf not in timeframes:
                timeframes.append(tf)
        if not timeframes:
            timeframes = list(frames.keys())

        if not full_tf:
            discovery_frames = frames.copy()
            if base_tf in discovery_frames and base_dataset is not None:
                rich_df = self._prepared_dataset_to_frame(
                    base_dataset,
                    fallback_frame=discovery_frames.get(base_tf),
                )
                if rich_df is None:
                    rich_df = discovery_frames.get(base_tf)
                discovery_frames[base_tf] = rich_df

            reference_df = discovery_frames.get(base_tf)
            aligned_frames = {}
            for tf in timeframes:
                if tf not in frames or reference_df is None:
                    continue
                local = _frame_copy(frames[tf])
                if local is None:
                    local = frames[tf]
                src_idx = _index_to_ns_generic(_frame_index(reference_df))
                tgt_idx = _index_to_ns_generic(_frame_index(local))
                tgt_n = _frame_len(local)
                for col in _frame_columns(reference_df):
                    low = str(col).lower()
                    if low in {"open", "high", "low", "close", "volume"}:
                        continue
                    if _frame_has_column(local, col):
                        continue
                    with contextlib.suppress(Exception):
                        src_vals = _frame_column_numpy(reference_df, col, dtype=np.float32)
                        aligned = _align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32)
                        if aligned is None:
                            aligned = _fit_len_array(src_vals, tgt_n, fill=0.0, dtype=np.float32)
                        _frame_set_column(local, col, aligned, dtype=np.float32)
                aligned_frames[tf] = local
            aligned_frames = self._inject_discovery_mixer_signals(
                aligned_frames if aligned_frames else discovery_frames,
                base_tf=base_tf,
                per_tf=False,
            )
            return aligned_frames, timeframes

        # Full per-TF feature engineering (slow but pure). Prefer Rust feature extraction when enabled.
        prefer_rust_discovery = self._discovery_rust_features_enabled()
        per_tf = {}
        for tf in timeframes:
            if tf not in frames:
                continue
            try:
                if base_dataset is not None and tf == base_tf:
                    rich_df = self._prepared_dataset_to_frame(base_dataset, fallback_frame=frames.get(tf))
                else:
                    tf_settings = self.settings.model_copy()
                    tf_settings.system.base_timeframe = tf
                    fe = FeatureEngineer(tf_settings)
                    rich_df = None
                    if prefer_rust_discovery and symbol:
                        ds_tf_rust = fe.prepare({}, news_features=None, symbol=symbol)
                        if self._dataset_row_count(ds_tf_rust) > 0:
                            rich_df = self._prepared_dataset_to_frame(ds_tf_rust, fallback_frame=frames.get(tf))
                    if rich_df is None:
                        ds_tf = fe.prepare(frames, news_features=news_feats, symbol=symbol)
                        rich_df = self._prepared_dataset_to_frame(ds_tf, fallback_frame=frames.get(tf))
                if rich_df is None:
                    continue
                per_tf[tf] = rich_df
            except Exception as exc:
                logger.warning(f"Discovery per-TF feature gen failed for {tf}: {exc}", exc_info=True)

        if not per_tf:
            return frames.copy(), timeframes

        all_cols: list[str] = []
        for df in per_tf.values():
            all_cols.extend(list(df.columns))
        # Preserve order, de-duplicate
        all_cols = list(dict.fromkeys(all_cols))

        aligned_frames = {}
        for tf, df in per_tf.items():
            aligned = _frame_copy(df)
            if aligned is None:
                continue
            n_rows = _frame_len(aligned)
            for col in all_cols:
                if _frame_has_column(aligned, col):
                    continue
                _frame_set_column(aligned, col, np.zeros(n_rows, dtype=np.float32), dtype=np.float32)
            orig = frames.get(tf)
            src_idx = _index_to_ns_generic(_frame_index(orig))
            tgt_idx = _index_to_ns_generic(_frame_index(aligned))
            for col in ["open", "high", "low", "close"]:
                if not _frame_has_column(orig, col):
                    continue
                with contextlib.suppress(Exception):
                    src_vals = _frame_column_numpy(orig, col, dtype=np.float64)
                    vals = _align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float64)
                    if vals is None:
                        vals = _fit_len_array(src_vals, n_rows, fill=0.0, dtype=np.float64)
                    _frame_set_column(aligned, col, vals, dtype=np.float64)
            aligned_frames[tf] = aligned

        aligned_frames = self._inject_discovery_mixer_signals(
            aligned_frames,
            base_tf=base_tf,
            per_tf=True,
        )
        return aligned_frames, timeframes

    def _inject_discovery_mixer_signals(
        self,
        frames: dict[str, Any],
        *,
        base_tf: str,
        per_tf: bool,
    ) -> dict[str, Any]:
        use_mixer = str(os.environ.get("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "1") or "1").strip().lower()
        if use_mixer not in {"1", "true", "yes", "on"}:
            return frames

        try:
            n_strategies = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_STRATEGIES", "24") or 24)
        except Exception:
            n_strategies = 24
        if n_strategies <= 0:
            return frames
        try:
            max_indicators = int(
                os.environ.get("FOREX_BOT_DISCOVERY_MIXER_MAX_INDICATORS", "0") or 0
            )
        except Exception:
            max_indicators = 0
        if max_indicators <= 0:
            try:
                max_indicators = int(
                    getattr(self.settings.models, "prop_search_max_indicators", 0) or 0
                )
            except Exception:
                max_indicators = 0
        if max_indicators <= 0:
            max_indicators = 3

        fb = None
        rust_bulk = False
        with contextlib.suppress(Exception):
            import forex_bindings as _fb  # type: ignore

            if hasattr(_fb, "talib_bulk_signals_ohlcv"):
                fb = _fb
                rust_bulk = True

        if rust_bulk and fb is not None:
            raw_pool = str(
                os.environ.get(
                    "FOREX_BOT_DISCOVERY_RUST_INDICATORS",
                    "RSI,ADX,MACD,ATR,NATR,EMA,SMA,CCI,ROC,MOM",
                )
                or "RSI,ADX,MACD,ATR,NATR,EMA,SMA,CCI,ROC,MOM"
            )
            pool = [s.strip().upper() for s in raw_pool.split(",") if str(s).strip()]
            if not pool:
                pool = ["RSI", "ADX", "MACD", "ATR", "NATR", "EMA", "SMA"]
            max_k = max(1, min(int(max_indicators), len(pool)))
            try:
                seed = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_SEED", "1337") or 1337)
            except Exception:
                seed = 1337
            rng = np.random.default_rng(seed)
            indicator_sets: list[list[str]] = []
            weight_sets: list[list[float]] = []
            long_thresholds: list[float] = []
            short_thresholds: list[float] = []
            for _ in range(n_strategies):
                k = int(rng.integers(1, max_k + 1))
                inds = rng.choice(np.asarray(pool, dtype=object), size=k, replace=False).tolist()
                indicator_sets.append([str(x).upper() for x in inds])
                weight_sets.append((0.5 + rng.random(k) * 1.0).astype(np.float64).tolist())
                long_thresholds.append(float(0.4 + rng.random() * 0.6))
                short_thresholds.append(float(-1.0 + rng.random() * 0.6))

            try:
                causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
            except Exception:
                causal_min_bars = 30
            causal_min_bars = max(2, causal_min_bars)

            def _rust_apply(df: Any) -> Any:
                if _frame_empty(df):
                    return df
                cols = {str(c).lower() for c in _frame_columns(df)}
                if not {"open", "high", "low", "close"}.issubset(cols):
                    return df
                local = _frame_copy(df)
                if local is None:
                    return df
                open_arr = _frame_column_numpy(local, "open", dtype=np.float64)
                high_arr = _frame_column_numpy(local, "high", dtype=np.float64)
                low_arr = _frame_column_numpy(local, "low", dtype=np.float64)
                close_arr = _frame_column_numpy(local, "close", dtype=np.float64)
                volume_arr = (
                    _frame_column_numpy(local, "volume", dtype=np.float64)
                    if bool(getattr(self.settings.system, "use_volume_features", False)) and _frame_has_column(local, "volume")
                    else None
                )
                try:
                    raw = fb.talib_bulk_signals_ohlcv(
                        open_arr,
                        high_arr,
                        low_arr,
                        close_arr,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        volume=volume_arr,
                        include_raw=False,
                        causal_min_bars=causal_min_bars,
                    )
                except TypeError:
                    raw = fb.talib_bulk_signals_ohlcv(
                        open_arr,
                        high_arr,
                        low_arr,
                        close_arr,
                        indicator_sets=indicator_sets,
                        weight_sets=weight_sets,
                        long_thresholds=long_thresholds,
                        short_thresholds=short_thresholds,
                        volume=volume_arr,
                        include_raw=False,
                    )
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim != 2:
                    return local
                if arr.shape[0] == len(indicator_sets) and arr.shape[1] == len(local):
                    arr = arr.T
                if arr.shape[0] != len(local):
                    return local
                width = min(arr.shape[1], len(indicator_sets))
                for idx in range(width):
                    col = f"tmx_sig_{idx}"
                    if _frame_has_column(local, col):
                        continue
                    _frame_set_column(local, col, np.asarray(arr[:, idx], dtype=np.float32), dtype=np.float32)
                return local

            try:
                if per_tf:
                    out = {}
                    for tf, df in frames.items():
                        out[tf] = _rust_apply(df)
                    logger.info("Discovery: Injected %s Rust mixer signals per TF.", n_strategies)
                    return out

                if base_tf not in frames:
                    return frames
                base_df = _rust_apply(frames[base_tf])
                sig_cols = [c for c in _frame_columns(base_df) if str(c).startswith("tmx_sig_")]
                out = {base_tf: base_df}
                for tf, df in frames.items():
                    if tf == base_tf:
                        continue
                    if not sig_cols:
                        out[tf] = df
                        continue
                    local = _frame_copy(df)
                    if local is None:
                        out[tf] = df
                        continue
                    src_idx = _index_to_ns_generic(_frame_index(base_df))
                    tgt_idx = _index_to_ns_generic(_frame_index(local))
                    tgt_n = _frame_len(local)
                    for col in sig_cols:
                        if _frame_has_column(local, col):
                            continue
                        with contextlib.suppress(Exception):
                            sig_src = _frame_column_numpy(base_df, col, dtype=np.float32)
                            aligned = _align_ffill_by_ns(src_idx, sig_src, tgt_idx, dtype=np.float32)
                            if aligned is None:
                                aligned = _fit_len_array(sig_src, tgt_n, fill=0.0, dtype=np.float32)
                            _frame_set_column(local, col, aligned, dtype=np.float32)
                    out[tf] = local
                logger.info("Discovery: Injected %s Rust mixer signals (base TF aligned).", len(sig_cols))
                return out
            except Exception as exc:
                logger.warning("Discovery: Rust mixer signal injection failed: %s", exc)
                return frames

        logger.warning("Discovery: Rust mixer backend unavailable; skipping mixer signals.")
        return frames

    @staticmethod
    def _parse_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except Exception:
            return None

    def _auto_feature_workers(self, symbol_count: int) -> int:
        symbol_count = max(1, int(symbol_count))
        cpu_total = max(1, os.cpu_count() or 1)
        cpu_reserve = self._parse_int_env("FOREX_BOT_CPU_RESERVE")
        if cpu_reserve is None:
            cpu_reserve = 1
        cpu_budget = max(1, cpu_total - max(0, int(cpu_reserve)))

        per_worker_gb = 6.0
        try:
            per_worker_gb = float(
                os.environ.get(
                    "FOREX_BOT_FEATURE_WORKER_GB_AUTO",
                    os.environ.get("FOREX_BOT_FEATURE_WORKER_GB", "6.0"),
                )
                or 6.0
            )
        except Exception:
            per_worker_gb = 6.0
        per_worker_gb = max(0.5, per_worker_gb)

        available_gb = None
        with contextlib.suppress(Exception):
            import psutil

            available_gb = float(psutil.virtual_memory().available) / float(1024**3)
        if available_gb is None:
            try:
                available_gb = float(os.environ.get("FOREX_BOT_RAM_GB", 0) or 0)
            except Exception:
                available_gb = 0.0
        if available_gb <= 0:
            available_gb = 16.0

        ram_workers = max(1, int(available_gb // per_worker_gb))
        return max(1, min(symbol_count, cpu_budget, ram_workers))

    def _infer_global_pool_cap_per_symbol(self, *, n_features: int, n_symbols: int) -> int | None:
        """
        Determine a safe per-symbol row cap for multi-symbol pooled training.

        Order of precedence:
        1) `FOREX_BOT_GLOBAL_MAX_ROWS_PER_SYMBOL` (int)
        2) `FOREX_BOT_GLOBAL_MAX_ROWS` (int) divided by symbol count
        3) Auto-fit to available RAM (conservative estimate)

        Set either env var to a value <= 0 to disable capping.
        """
        n_symbols = max(1, int(n_symbols))
        n_features = max(1, int(n_features))

        explicit_per_symbol = self._parse_int_env("FOREX_BOT_GLOBAL_MAX_ROWS_PER_SYMBOL")
        if explicit_per_symbol is None:
            explicit_per_symbol = int(
                getattr(getattr(self.settings, "models", None), "global_max_rows_per_symbol", 0) or 0
            )
        if explicit_per_symbol is not None and explicit_per_symbol <= 0:
            return None
        if explicit_per_symbol is not None and explicit_per_symbol > 0:
            return int(explicit_per_symbol)

        explicit_total = self._parse_int_env("FOREX_BOT_GLOBAL_MAX_ROWS")
        if explicit_total is None:
            explicit_total = int(getattr(getattr(self.settings, "models", None), "global_max_rows", 0) or 0)
        if explicit_total is not None and explicit_total <= 0:
            return None
        if explicit_total is not None and explicit_total > 0:
            return max(1, int(explicit_total) // n_symbols)

        # Auto cap based on available RAM. Keep conservative to avoid OOM on Windows.
        try:
            import psutil

            available = float(psutil.virtual_memory().available)
        except Exception:
            return None

        try:
            mem_frac = float(os.environ.get("FOREX_BOT_GLOBAL_POOL_MEM_FRAC", "0.10") or 0.10)
        except Exception:
            mem_frac = 0.10
        mem_frac = float(min(0.40, max(0.05, mem_frac)))

        try:
            overhead = float(os.environ.get("FOREX_BOT_GLOBAL_POOL_OVERHEAD", "3.0") or 3.0)
        except Exception:
            overhead = 3.0
        overhead = float(min(10.0, max(1.5, overhead)))

        bytes_per_row = float(n_features) * 4.0 * overhead  # float32 payload + tabular overhead factor
        budget = available * mem_frac
        total_rows = int(budget // max(bytes_per_row, 1.0))
        per_symbol = max(1, total_rows // n_symbols)

        try:
            floor = int(os.environ.get("FOREX_BOT_GLOBAL_MIN_ROWS_PER_SYMBOL", "50000") or 50000)
        except Exception:
            floor = 50000
        per_symbol = max(1, max(floor, per_symbol))
        return int(per_symbol)

    @staticmethod
    def _tail_dataset(ds: PreparedDataset, rows: int) -> PreparedDataset:
        if rows <= 0:
            return ds
        try:
            n = len(ds.X)
        except Exception:
            return ds
        if n <= rows:
            return ds

        def _tail(obj):
            try:
                if _is_dataframe(obj):
                    n_obj = len(obj)
                    start = max(0, int(n_obj) - int(rows))
                    return _slice_rows_range(obj, start, n_obj)
                if _is_frame_like(obj):
                    n_obj = _frame_len(obj)
                    return _slice_rows_range(obj, max(0, n_obj - rows), n_obj)
                return obj[-rows:]
            except Exception:
                return obj

        X = _tail(ds.X)
        y = _tail(ds.y)
        labels = _tail(ds.labels) if ds.labels is not None else None
        meta = _tail(ds.metadata) if (_is_dataframe(ds.metadata) or _is_frame_like(ds.metadata)) else ds.metadata
        return PreparedDataset(
            X=X,
            y=y,
            index=getattr(X, "index", None),
            feature_names=list(getattr(X, "columns", ds.feature_names)),
            metadata=meta,
            labels=labels,
        )

    @staticmethod
    def _pair_corr_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_PAIR_CORR_ENABLED", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _shift_with_lag(values: np.ndarray, lag: int) -> np.ndarray:
        src = np.asarray(values, dtype=np.float32).reshape(-1)
        n = src.shape[0]
        out = np.zeros(n, dtype=np.float32)
        if n <= 0:
            return out
        lag_i = max(1, int(lag))
        if lag_i >= n:
            return out
        out[lag_i:] = src[:-lag_i]
        return out

    @staticmethod
    def _align_values_by_timestamp(src_idx: np.ndarray, src_vals: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
        out = _align_exact_by_ns(
            np.asarray(src_idx, dtype=np.int64).reshape(-1),
            np.asarray(src_vals, dtype=np.float32).reshape(-1),
            np.asarray(target_idx, dtype=np.int64).reshape(-1),
            dtype=np.float32,
            fill=0.0,
        )
        if out is None:
            return np.zeros(np.asarray(target_idx).reshape(-1).size, dtype=np.float32)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _align_values_ffill_by_timestamp(
        src_idx: np.ndarray,
        src_vals: np.ndarray,
        target_idx: np.ndarray,
        *,
        default: float = 0.0,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        out = _align_ffill_by_ns(
            np.asarray(src_idx, dtype=np.int64).reshape(-1),
            np.asarray(src_vals, dtype=np.float64).reshape(-1),
            np.asarray(target_idx, dtype=np.int64).reshape(-1),
            dtype=dtype,
            fill=float(default),
        )
        if out is None:
            return np.full(np.asarray(target_idx).reshape(-1).size, float(default), dtype=dtype)
        return np.asarray(out, dtype=dtype).reshape(-1)

    @staticmethod
    def _rolling_corr_numpy(a: np.ndarray, b: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        a64 = np.asarray(a, dtype=np.float64).reshape(-1)
        b64 = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(a64.size, b64.size))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        a64 = a64[:n]
        b64 = b64[:n]
        out = np.zeros(n, dtype=np.float32)
        w = max(2, int(window))
        mp = max(2, int(min_periods))

        cs_a = np.zeros(n + 1, dtype=np.float64)
        cs_b = np.zeros(n + 1, dtype=np.float64)
        cs_aa = np.zeros(n + 1, dtype=np.float64)
        cs_bb = np.zeros(n + 1, dtype=np.float64)
        cs_ab = np.zeros(n + 1, dtype=np.float64)
        np.cumsum(a64, out=cs_a[1:])
        np.cumsum(b64, out=cs_b[1:])
        np.cumsum(a64 * a64, out=cs_aa[1:])
        np.cumsum(b64 * b64, out=cs_bb[1:])
        np.cumsum(a64 * b64, out=cs_ab[1:])

        for i in range(n):
            start = max(0, i - w + 1)
            m = i - start + 1
            if m < mp:
                continue
            sa = cs_a[i + 1] - cs_a[start]
            sb = cs_b[i + 1] - cs_b[start]
            saa = cs_aa[i + 1] - cs_aa[start]
            sbb = cs_bb[i + 1] - cs_bb[start]
            sab = cs_ab[i + 1] - cs_ab[start]
            ma = sa / m
            mb = sb / m
            var_a = (saa / m) - (ma * ma)
            var_b = (sbb / m) - (mb * mb)
            if var_a <= 1e-12 or var_b <= 1e-12:
                continue
            cov = (sab / m) - (ma * mb)
            out[i] = float(cov / np.sqrt(var_a * var_b))
        return out

    def _numpy_dataset_returns(self, ds: PreparedDataset) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            x_np = np.asarray(ds.X, dtype=np.float32)
        except Exception:
            return None
        if x_np.ndim != 2 or x_np.shape[0] <= 1:
            return None

        names = [str(c).strip().lower() for c in (ds.feature_names or [])]
        n_rows = int(x_np.shape[0])
        idx_ns = self._index_to_ns(getattr(ds, "index", None))
        if idx_ns is None or len(idx_ns) != n_rows:
            idx_ns = np.arange(n_rows, dtype=np.int64)
        else:
            idx_ns = np.asarray(idx_ns, dtype=np.int64).reshape(-1)

        ret = None
        if names:
            ret_i = None
            for key in ("returns", "ret", "return"):
                with contextlib.suppress(ValueError):
                    ret_i = names.index(key)
                    break
            if ret_i is not None:
                ret = x_np[:, ret_i].astype(np.float32, copy=False)

            if ret is None:
                close_i = None
                for key in ("close", "price_close", "mid_close"):
                    with contextlib.suppress(ValueError):
                        close_i = names.index(key)
                        break
                if close_i is not None:
                    close = x_np[:, close_i].astype(np.float32, copy=False)
                    ret = np.zeros(n_rows, dtype=np.float32)
                    prev = np.where(np.abs(close[:-1]) > 1e-9, close[:-1], 1e-9)
                    ret[1:] = (close[1:] - close[:-1]) / prev

        if ret is None:
            meta = getattr(ds, "metadata", None)
            if (_is_dataframe(meta) or _is_frame_like(meta)) and _frame_has_column(meta, "close"):
                try:
                    close_src = _frame_column_numpy(meta, "close", dtype=np.float32)
                    meta_idx = self._index_to_ns(_frame_index(meta))
                    if meta_idx is not None and len(meta_idx) == close_src.shape[0]:
                        close = self._align_values_by_timestamp(meta_idx, close_src, idx_ns)
                    elif close_src.shape[0] == n_rows:
                        close = close_src.astype(np.float32, copy=False)
                    else:
                        close = None
                    if close is not None and close.shape[0] == n_rows:
                        ret = np.zeros(n_rows, dtype=np.float32)
                        prev = np.where(np.abs(close[:-1]) > 1e-9, close[:-1], 1e-9)
                        ret[1:] = (close[1:] - close[:-1]) / prev
                except Exception:
                    ret = None

        if ret is None:
            return None
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return idx_ns, ret

    def _inject_cross_pair_context_numpy(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        *,
        window: int,
        lag: int,
        max_peers: int,
        min_overlap: int,
        static_rows: int,
    ) -> list[tuple[str, PreparedDataset]]:
        returns_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sym, ds in datasets:
            parsed = self._numpy_dataset_returns(ds)
            if parsed is not None:
                returns_map[str(sym)] = parsed

        if len(returns_map) < 2:
            logger.info("[GLOBAL CORR] Skipping pair-correlation features (insufficient NumPy return series).")
            return datasets

        feature_suffixes = [
            "pair_peer_ret_mean",
            "pair_peer_ret_std",
            "pair_peer_abs_mean",
            "pair_corr_mean",
            "pair_corr_abs_mean",
            "pair_corr_max",
            "pair_corr_min",
            "pair_lead_ret",
            "pair_lead_corr",
            "pair_divergence",
            "pair_relative_strength",
            "pair_static_corr_mean",
            "pair_static_corr_abs_max",
        ]

        patched: list[tuple[str, PreparedDataset]] = []
        for sym, ds in datasets:
            sym_key = str(sym)
            sym_parsed = returns_map.get(sym_key)
            if sym_parsed is None:
                patched.append((sym, ds))
                continue
            sym_idx, sym_ret = sym_parsed

            peer_rank: list[tuple[str, float, float]] = []
            for peer_sym, (peer_idx, peer_ret) in returns_map.items():
                if peer_sym == sym_key:
                    continue
                peer_aligned = self._align_values_by_timestamp(peer_idx, peer_ret, sym_idx)
                if peer_aligned.shape[0] != sym_ret.shape[0]:
                    continue
                if sym_ret.shape[0] > static_rows:
                    a = sym_ret[-static_rows:]
                    b = peer_aligned[-static_rows:]
                else:
                    a = sym_ret
                    b = peer_aligned
                mask = np.isfinite(a) & np.isfinite(b)
                overlap = int(np.sum(mask))
                if overlap < min_overlap:
                    continue
                with contextlib.suppress(Exception):
                    corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                    if np.isfinite(corr):
                        peer_rank.append((peer_sym, abs(corr), corr))
            if peer_rank:
                order = _rust_rank_scores_desc(np.asarray([item[1] for item in peer_rank], dtype=np.float64))
                if order is not None:
                    peer_rank = [peer_rank[int(i)] for i in order.tolist()]
                else:
                    peer_rank.sort(key=lambda item: item[1], reverse=True)
            peers = [p for p, _abs, _corr in peer_rank[:max_peers]]
            if not peers:
                patched.append((sym, ds))
                continue

            x_np = np.asarray(ds.X, dtype=np.float32)
            n_rows = int(x_np.shape[0])
            if n_rows <= 0:
                patched.append((sym, ds))
                continue

            sym_hist = self._shift_with_lag(sym_ret, lag)
            peer_hist = np.zeros((n_rows, len(peers)), dtype=np.float32)
            corr_hist = np.zeros((n_rows, len(peers)), dtype=np.float32)
            min_periods = max(20, window // 4)

            for j, p in enumerate(peers):
                p_idx, p_ret = returns_map[p]
                p_aligned = self._align_values_by_timestamp(p_idx, p_ret, sym_idx)
                p_hist = self._shift_with_lag(p_aligned, lag)
                peer_hist[:, j] = p_hist
                corr_hist[:, j] = self._rolling_corr_numpy(sym_hist, p_hist, window=window, min_periods=min_periods)

            lead_peer = peers[0]
            lead_idx = peers.index(lead_peer)
            static_corrs = [c for _p, _abs, c in peer_rank[:max_peers]]

            with contextlib.suppress(Exception):
                features = np.column_stack(
                    [
                        np.mean(peer_hist, axis=1),
                        np.std(peer_hist, axis=1),
                        np.mean(np.abs(peer_hist), axis=1),
                        np.mean(corr_hist, axis=1),
                        np.mean(np.abs(corr_hist), axis=1),
                        np.max(corr_hist, axis=1),
                        np.min(corr_hist, axis=1),
                        peer_hist[:, lead_idx],
                        corr_hist[:, lead_idx],
                        sym_hist - np.mean(peer_hist, axis=1),
                        sym_hist - peer_hist[:, lead_idx],
                        np.full(n_rows, float(np.mean(static_corrs)) if static_corrs else 0.0, dtype=np.float32),
                        np.full(
                            n_rows,
                            float(max(abs(v) for v in static_corrs)) if static_corrs else 0.0,
                            dtype=np.float32,
                        ),
                    ]
                ).astype(np.float32, copy=False)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                x_aug = np.concatenate([x_np, features], axis=1)
                patched.append(
                    (
                        sym,
                        PreparedDataset(
                            X=x_aug,
                            y=ds.y,
                            index=ds.index,
                            feature_names=list(ds.feature_names) + feature_suffixes,
                            metadata=ds.metadata,
                            labels=ds.labels if ds.labels is not None else ds.y,
                        ),
                    )
                )
                logger.info(
                    "[GLOBAL CORR] %s: added NumPy pair-context features using peers=%s (window=%s, lag=%s).",
                    sym,
                    ",".join(peers),
                    window,
                    lag,
                )
                continue

            patched.append((sym, ds))
        return patched

    def _inject_cross_pair_context(
        self,
        datasets: list[tuple[str, PreparedDataset]],
    ) -> list[tuple[str, PreparedDataset]]:
        """
        Add cross-symbol correlation context for pooled multi-symbol training.
        Features are lagged to avoid same-bar leakage.
        """
        if len(datasets) < 2 or not self._pair_corr_enabled():
            return datasets

        window = self._parse_int_env("FOREX_BOT_PAIR_CORR_WINDOW") or 240
        window = max(16, int(window))
        lag = self._parse_int_env("FOREX_BOT_PAIR_CORR_LAG") or 1
        lag = max(1, int(lag))
        max_peers = self._parse_int_env("FOREX_BOT_PAIR_CORR_MAX_PEERS")
        if max_peers is None:
            max_peers = min(4, len(datasets) - 1)
        max_peers = max(1, min(int(max_peers), len(datasets) - 1))
        min_overlap = self._parse_int_env("FOREX_BOT_PAIR_CORR_MIN_OVERLAP")
        if min_overlap is None:
            min_overlap = max(50, window)
        min_overlap = max(20, int(min_overlap))
        static_rows = self._parse_int_env("FOREX_BOT_PAIR_CORR_STATIC_ROWS") or 200_000
        static_rows = max(5_000, int(static_rows))
        return self._inject_cross_pair_context_numpy(
            datasets,
            window=window,
            lag=lag,
            max_peers=max_peers,
            min_overlap=min_overlap,
            static_rows=static_rows,
        )

    async def train(self, optimize: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Run the full training pipeline for the single active symbol."""
        symbol = self.settings.system.symbol
        logger.info(f"Starting training for {symbol}...")
        logger.info("Frame-native mode: loading dataset from Rust features backend.")
        dataset = self.feature_engineer.prepare({}, news_features=None, symbol=symbol)
        if getattr(dataset, "X", None) is None or len(dataset.X) <= 0:
            logger.error("Frame-native mode: no dataset rows available for %s.", symbol)
            return

        await asyncio.to_thread(
            self.trainer.train_all,
            dataset,
            optimize,
            stop_event,
            None,
            None,
        )

        logger.info("Training cycle complete.")
        self._maybe_stop_ray()

    async def train_incremental_all(self, optimize: bool = False, stop_event: asyncio.Event | None = None) -> None:
        """
        Best-effort incremental retraining for the active symbol.
        Runs in a thread to avoid blocking the event loop when triggered from the trading loop.
        """
        symbol = self.settings.system.symbol
        logger.info(f"Starting incremental retraining for drift recovery on {symbol}...")

        try:
            frames = await self.data_loader.get_training_data(symbol)
            if not frames:
                raise RuntimeError("No training data available for incremental retrain")

            dataset = self.feature_engineer.prepare(frames, symbol=symbol)

            # Offload synchronous training work to a thread to keep async loop responsive
            await asyncio.to_thread(self.trainer.train_incremental, dataset, symbol, optimize, stop_event)
            logger.info("Incremental retraining finished.")
        except Exception as exc:
            logger.error(f"Incremental retraining failed: {exc}", exc_info=True)
            raise

    async def train_global(
        self, symbols: list[str], optimize: bool = True, stop_event: asyncio.Event | None = None
    ) -> None:
        """Train one global model across all provided symbols."""
        logger.info(f"Starting global training for symbols: {symbols}")
        await self._train_global_frame_native(symbols, optimize, stop_event)
        self._maybe_stop_ray()

    async def _train_global_frame_native(
        self,
        symbols: list[str],
        optimize: bool,
        stop_event: asyncio.Event | None,
    ) -> None:
        datasets: list[tuple[str, PreparedDataset]] = []
        requested_workers = self._parse_int_env("FOREX_BOT_FEATURE_WORKERS")
        if requested_workers is not None and requested_workers > 0:
            max_workers = max(1, min(len(symbols), int(requested_workers)))
        else:
            max_workers = self._auto_feature_workers(len(symbols))
        max_workers = max(1, min(max_workers, len(symbols)))

        async def _prepare_single(sym: str) -> tuple[str, PreparedDataset | None]:
            if stop_event and stop_event.is_set():
                return sym, None
            try:
                ds = await asyncio.to_thread(
                    self.feature_engineer.prepare,
                    {},
                    news_features=None,
                    symbol=sym,
                )
            except Exception as exc:
                logger.warning("Frame-native global: failed to prepare %s: %s", sym, exc)
                return sym, None
            return sym, ds

        if max_workers > 1 and len(symbols) > 1:
            logger.info(
                "Frame-native global: preparing %s symbols with up to %s workers.",
                len(symbols),
                max_workers,
            )
            semaphore = asyncio.Semaphore(max_workers)

            async def _prepare_single_limited(sym: str) -> tuple[str, PreparedDataset | None]:
                async with semaphore:
                    return await _prepare_single(sym)

            prepared = await asyncio.gather(*[_prepare_single_limited(sym) for sym in symbols])
        else:
            prepared = []
            for sym in symbols:
                if stop_event and stop_event.is_set():
                    break
                prepared.append(await _prepare_single(sym))

        for sym, ds in prepared:
            if ds is None:
                continue
            if getattr(ds, "X", None) is None or len(ds.X) <= 0:
                logger.warning("Frame-native global: empty dataset for %s; skipping.", sym)
                continue
            datasets.append((sym, ds))
        if not datasets:
            logger.error("Frame-native global: no datasets prepared.")
            return
        await self._train_global_from_datasets(
            datasets,
            [s for s, _ in datasets],
            optimize,
            stop_event,
            exclude_models=None,
        )

    def _build_news_features(self, analyzer: Any, symbol: str, frames: dict[str, Any]) -> Any | None:
        """Best-effort build of time-aligned news features for a symbol using the local news DB."""
        if self._rust_only_enabled():
            return None
        if analyzer is None:
            return None
        try:
            currencies = [symbol[:3], symbol[3:]] if isinstance(symbol, str) and len(symbol) == 6 else []
            base_tf = self.settings.system.base_timeframe
            base_df = frames.get(base_tf)
            if base_df is None:
                base_df = frames.get("M1")
            if base_df is None or not hasattr(base_df, "empty") or base_df.empty:
                return None

            base_idx, start_ts, end_ts = self._base_index_and_bounds(base_df, coerce_datetime_index=True)
            if base_idx is None or start_ts is None or end_ts is None:
                return None

            # Optional: rescore existing archive/DB events so training uses better sentiment/confidence.
            if bool(getattr(self.settings.news, "auto_rescore_enabled", False)):
                try:
                    days = int(getattr(self.settings.news, "auto_rescore_days", 30) or 0)
                    start_r = start_ts
                    if days > 0:
                        start_r = max(start_ts, end_ts - timedelta(days=days))
                    max_events = int(getattr(self.settings.news, "auto_rescore_max_events", 200) or 200)
                    only_missing = bool(getattr(self.settings.news, "auto_rescore_only_missing", True))
                    rescored = analyzer.rescore_existing_events(
                        symbol,
                        start_r,
                        end_ts,
                        max_events=max_events,
                        only_missing=only_missing,
                    )
                    # Safe check to prevent "The truth value of a DataFrame is ambiguous" error
                    has_rescored = False
                    count_str = "0"

                    if rescored is not None:
                        if isinstance(rescored, int) and rescored > 0:
                            has_rescored = True
                            count_str = str(rescored)
                        elif isinstance(rescored, list) and len(rescored) > 0:
                            has_rescored = True
                            count_str = str(len(rescored))
                        elif hasattr(rescored, "empty"):
                            # Handle DataFrame/Series
                            if not rescored.empty:
                                has_rescored = True
                                count_str = str(len(rescored))
                        elif hasattr(rescored, "size") and getattr(rescored, "size", 0) > 0:
                            # Handle numpy array or similar
                            has_rescored = True
                            count_str = str(rescored.size)
                        else:
                            try:
                                if rescored:
                                    has_rescored = True
                                    count_str = "some"
                            except ValueError:
                                # "The truth value of ... is ambiguous"
                                has_rescored = True  # Assume if it's ambiguous, it's a non-empty container
                                count_str = "unknown"

                    if has_rescored:
                        logger.info(f"Rescored {count_str} news events for {symbol} (lookback={days}d).")
                except Exception as exc:
                    logger.debug(f"News rescore skipped for {symbol}: {exc}")

            events = analyzer.db.fetch_events(start_ts, end_ts, currencies=currencies)

            # Safe check for events (list or DataFrame)
            has_events = False
            if events is not None:
                if isinstance(events, list):
                    has_events = len(events) > 0
                elif hasattr(events, "empty"):
                    has_events = not events.empty
                else:
                    try:
                        has_events = len(events) > 0
                    except Exception:
                        has_events = bool(events)

            if not has_events:
                return None

            nf = analyzer.build_features(events, base_idx)

            # Safe check for features (DataFrame)
            has_features = False
            if nf is not None:
                if hasattr(nf, "empty"):
                    has_features = not nf.empty
                else:
                    try:
                        has_features = len(nf) > 0
                    except Exception:
                        has_features = bool(nf)

            return nf if has_features else None
        except Exception as exc:
            logger.debug(f"News feature build skipped for {symbol}: {exc}")
            return None

    @staticmethod
    def _aggregate_metrics(
        metrics_list: list[dict[str, Any]],
        *,
        numeric_keys: list[str],
        bool_any_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if not metrics_list:
            return {}

        def _num(key: str) -> list[float]:
            vals: list[float] = []
            for m in metrics_list:
                v = m.get(key)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            return vals

        agg: dict[str, Any] = {}
        for key in numeric_keys:
            vals = _num(key)
            if not vals:
                continue
            agg[key] = float(np.mean(vals))

        for key in bool_any_keys or []:
            agg[key] = any(bool(m.get(key)) for m in metrics_list)
        return agg

    def _align_global_feature_space(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        *,
        prefer_numpy: bool = False,
    ) -> tuple[list[str], list[tuple[str, PreparedDataset]]]:
        all_cols: set[str] = set()
        for _, d in datasets:
            try:
                if _is_dataframe(d.X):
                    all_cols.update([str(c) for c in d.X.columns])
                elif _is_frame_like(d.X):
                    names = _frame_columns(d.X)
                    if names:
                        all_cols.update([str(c) for c in names])
                    else:
                        x_np, inferred = _frame_to_2d_float32(d.X, feature_names=list(getattr(d, "feature_names", []) or []))
                        if inferred:
                            all_cols.update(inferred)
                        elif x_np.ndim == 2:
                            all_cols.update([f"f{i}" for i in range(x_np.shape[1])])
                else:
                    names = list(getattr(d, "feature_names", []) or [])
                    if names:
                        all_cols.update([str(c) for c in names])
                    else:
                        x_np = np.asarray(d.X)
                        if x_np.ndim == 2:
                            all_cols.update([f"f{i}" for i in range(x_np.shape[1])])
            except Exception:
                continue
        cols = sorted(all_cols)
        col_to_idx = {str(col): i for i, col in enumerate(cols)}

        aligned: list[tuple[str, PreparedDataset]] = []
        for sym, d in datasets:
            try:
                row_index: Any
                if _is_dataframe(d.X):
                    row_index = d.X.index
                    rows = len(d.X)
                    x_src = d.X.to_numpy(dtype=np.float32, copy=False)
                    src_names = [str(c) for c in d.X.columns]
                    src_cols, dst_cols = _column_index_mapping(src_names, col_to_idx)
                    x_aligned = _align_feature_matrix(
                        x_src,
                        src_cols,
                        dst_cols,
                        dst_width=len(cols),
                    )
                    if prefer_numpy:
                        X = x_aligned
                        idx_ns = self._index_to_ns(row_index)
                        if idx_ns is not None and idx_ns.size == rows:
                            X_index = idx_ns
                        else:
                            X_index = np.arange(rows, dtype=np.int64)
                    else:
                        X = _make_dataframe(x_aligned, columns=cols, index=row_index)
                        row_index = _frame_index(X)
                        if row_index is None:
                            row_index = d.X.index
                        X_index = row_index
                elif _is_frame_like(d.X):
                    x_src, src_names = _frame_to_2d_float32(
                        d.X,
                        feature_names=list(getattr(d, "feature_names", []) or []),
                    )
                    rows = int(x_src.shape[0])
                    src_cols, dst_cols = _column_index_mapping([str(name) for name in src_names], col_to_idx)
                    X = _align_feature_matrix(
                        x_src,
                        src_cols,
                        dst_cols,
                        dst_width=len(cols),
                    )
                    row_index = _frame_index(d.X)
                    if row_index is None:
                        row_index = getattr(d, "index", np.arange(rows, dtype=np.int64))
                    idx_ns = self._index_to_ns(row_index)
                    if idx_ns is None or idx_ns.size != rows:
                        idx_ns = np.arange(rows, dtype=np.int64)
                    X_index = idx_ns
                    row_index = X_index
                else:
                    x_src = np.asarray(d.X, dtype=np.float32)
                    if x_src.ndim != 2:
                        raise ValueError("non-2d feature matrix")
                    rows = int(x_src.shape[0])
                    names = list(getattr(d, "feature_names", []) or [])
                    if len(names) != x_src.shape[1]:
                        names = [f"f{i}" for i in range(x_src.shape[1])]
                    src_cols, dst_cols = _column_index_mapping([str(name) for name in names], col_to_idx)
                    X = _align_feature_matrix(
                        x_src,
                        src_cols,
                        dst_cols,
                        dst_width=len(cols),
                    )
                    idx_ns = self._index_to_ns(getattr(d, "index", np.arange(rows, dtype=np.int64)))
                    if idx_ns is None or idx_ns.size != rows:
                        idx_ns = np.arange(rows, dtype=np.int64)
                    X_index = idx_ns
                    row_index = X_index

                y_raw = d.y
                if _is_series(y_raw):
                    y_np = _series_like_to_int8(
                        y_raw,
                        row_index=row_index if len(y_raw) != rows else None,
                        n_rows=rows,
                    )
                    if prefer_numpy or not _is_dataframe(d.X):
                        y = y_np
                    else:
                        y = _make_series(y_np, index=row_index, dtype=np.int8)
                else:
                    y = np.asarray(y_raw, dtype=np.int8).reshape(-1)
                    if y.shape[0] != rows:
                        raise ValueError(f"label length mismatch: {y.shape[0]} != {rows}")
                meta = d.metadata
                if _is_dataframe(meta):
                    if len(meta) != rows:
                        meta = _slice_rows_range(meta, 0, rows)
                elif _is_frame_like(meta):
                    if len(meta) != rows:
                        meta = _slice_rows_range(meta, 0, rows)

                aligned.append(
                    (
                        sym,
                        PreparedDataset(
                            X=X,
                            y=y,
                            index=X_index,
                            feature_names=cols,
                            metadata=meta,
                            labels=y,
                        ),
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to align dataset for {sym}: {exc}")
        return cols, aligned

    @staticmethod
    def _index_to_ns(index: Any) -> np.ndarray | None:
        return _index_to_ns_int64(index)

    @staticmethod
    def _ns_bounds_to_py_utc(ns_values: np.ndarray | None) -> tuple[datetime, datetime] | None:
        if ns_values is None:
            return None
        try:
            arr = np.asarray(ns_values, dtype=np.int64).reshape(-1)
            if arr.size <= 0:
                return None
            nat = np.iinfo(np.int64).min
            valid = arr[arr != nat]
            if valid.size <= 0:
                return None
            start_ns = int(np.min(valid))
            end_ns = int(np.max(valid))
            start_ts = datetime.fromtimestamp(start_ns / 1_000_000_000.0, tz=timezone.utc)
            end_ts = datetime.fromtimestamp(end_ns / 1_000_000_000.0, tz=timezone.utc)
            return start_ts, end_ts
        except Exception:
            return None

    def _base_index_and_bounds(
        self,
        base_df: Any,
        *,
        coerce_datetime_index: bool = False,
    ) -> tuple[Any | None, datetime | None, datetime | None]:
        if base_df is None or not hasattr(base_df, "empty") or base_df.empty:
            return None, None, None
        idx_source = base_df["timestamp"] if hasattr(base_df, "columns") and "timestamp" in base_df.columns else base_df.index
        idx_ns = self._index_to_ns(idx_source)
        bounds = self._ns_bounds_to_py_utc(idx_ns)
        start_ts = bounds[0] if bounds is not None else None
        end_ts = bounds[1] if bounds is not None else None
        if not coerce_datetime_index:
            return None, start_ts, end_ts
        if idx_ns is None:
            return None, start_ts, end_ts
        try:
            idx_dt = idx_ns.astype("datetime64[ns]")
        except Exception:
            return None, start_ts, end_ts
        return idx_dt, start_ts, end_ts

    @staticmethod
    def _month_day_indices_from_index(index_like: Any) -> tuple[np.ndarray, np.ndarray]:
        rust = _rust_time_index_arrays(index_like)
        if rust is not None:
            _unix_ms, month_idx, day_idx = rust
            return month_idx, day_idx
        idx_ns = TrainingService._index_to_ns(index_like)
        if idx_ns is None or idx_ns.size <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        dt = idx_ns.astype("datetime64[ns]")
        month_idx = dt.astype("datetime64[M]").astype(np.int64, copy=False)
        day_idx = dt.astype("datetime64[D]").astype(np.int64, copy=False)
        return month_idx, day_idx

    def _split_global_train_eval(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        *,
        train_ratio: float,
        embargo_bars: int,
        min_train_rows: int = 1000,
        min_eval_rows: int = 500,
    ) -> tuple[list[tuple[str, PreparedDataset]], dict[str, PreparedDataset], dict[str, Any]]:
        # Compute a global time cutoff so no symbol trains on "future" relative to any other.
        times: list[np.ndarray] = []
        for _, d in datasets:
            idx_ns = self._index_to_ns(d.index)
            if idx_ns is not None and idx_ns.size > 0:
                times.append(idx_ns)
        if not times:
            raise RuntimeError("Global split failed: no timestamps found.")

        all_times = np.concatenate(times)
        all_times.sort()
        first_ns = int(all_times[0])
        last_ns = int(all_times[-1])

        eval_from_raw = str(os.environ.get("FOREX_BOT_GLOBAL_EVAL_FROM", "") or "").strip()
        try:
            eval_years = float(os.environ.get("FOREX_BOT_GLOBAL_EVAL_YEARS", "0") or 0.0)
        except Exception:
            eval_years = 0.0
        eval_years = max(0.0, float(eval_years))

        cutoff_mode = "ratio"
        cutoff_ns: int
        if eval_from_raw:
            try:
                parsed = np.datetime64(eval_from_raw).astype("datetime64[ns]")
                cutoff_ns = int(parsed.astype(np.int64))
                cutoff_mode = "from"
            except Exception:
                logger.warning(
                    "Invalid FOREX_BOT_GLOBAL_EVAL_FROM=%r; falling back to ratio split.",
                    eval_from_raw,
                )
                cut_i = int(max(0, min(len(all_times) - 1, int(len(all_times) * train_ratio) - 1)))
                cutoff_ns = int(all_times[cut_i])
        elif eval_years > 0.0:
            year_ns = int(float(eval_years) * 365.2425 * 24.0 * 3600.0 * 1_000_000_000.0)
            cutoff_ns = int(last_ns - year_ns)
            cutoff_mode = "years"
        else:
            cut_i = int(max(0, min(len(all_times) - 1, int(len(all_times) * train_ratio) - 1)))
            cutoff_ns = int(all_times[cut_i])

        # Clamp to observed span to avoid impossible cutoffs.
        if cutoff_ns < first_ns:
            cutoff_ns = first_ns
        elif cutoff_ns > last_ns:
            cutoff_ns = last_ns

        cutoff_dt = datetime.fromtimestamp(cutoff_ns / 1_000_000_000.0, tz=timezone.utc)

        train_parts: list[tuple[str, PreparedDataset]] = []
        eval_map: dict[str, PreparedDataset] = {}
        split_meta: dict[str, Any] = {
            "cutoff": cutoff_dt.isoformat(),
            "cutoff_mode": cutoff_mode,
            "eval_years": float(eval_years),
            "eval_from": eval_from_raw,
            "embargo_bars": int(embargo_bars),
            "per_symbol": {},
        }

        for sym, d in datasets:
            X = d.X
            is_df = _is_dataframe(X)
            is_frame = _is_frame_like(X) and not is_df
            meta = d.metadata
            y_source = d.y
            reordered = False
            n = len(X)
            if n == 0:
                continue

            idx = _frame_index(X) if (is_df or is_frame) else np.asarray(d.index)
            if idx is None:
                idx = np.asarray(d.index)
            idx_ns = self._index_to_ns(idx)
            if idx_ns is None or idx_ns.size != n:
                logger.warning("Skipping %s: invalid index for global split.", sym)
                continue
            order = _sorted_time_order(idx, n)
            if order is not None:
                reordered = True
                X = _slice_rows_positions(X, order)
                if meta is not None:
                    meta = _slice_rows_positions(meta, order)
                if _is_series(y_source):
                    y_source = _slice_rows_positions(y_source, order)
                else:
                    try:
                        order_arr = np.asarray(order, dtype=np.int64)
                        y_source = np.asarray(y_source).reshape(-1)[order_arr]
                    except Exception:
                        y_source = d.y
                idx = _frame_index(X) if (is_df or is_frame) else np.asarray(d.index)[order_arr]
                idx_ns = idx_ns[order_arr]
            try:
                cut_right = int(np.searchsorted(idx_ns, cutoff_ns, side="right"))
            except Exception:
                cut_right = int(n * train_ratio)

            train_end = max(0, cut_right - max(0, int(embargo_bars)))
            eval_start = cut_right

            if train_end < min_train_rows or (n - eval_start) < min_eval_rows:
                # Fallback to per-symbol ratio split if the global cutoff is too skewed for this symbol.
                cut_right = int(n * train_ratio)
                train_end = max(0, cut_right - max(0, int(embargo_bars)))
                eval_start = cut_right

            if train_end < max(100, min_train_rows // 5) or (n - eval_start) < max(200, min_eval_rows // 3):
                logger.warning(
                    f"Skipping {sym}: insufficient rows after split (train={train_end}, eval={n - eval_start})."
                )
                continue

            if _is_series(y_source):
                y_row_index = idx if (len(y_source) != n or reordered) else None
                y_full = _series_like_to_int8(y_source, row_index=y_row_index, n_rows=n)
            else:
                y_full = np.asarray(y_source, dtype=np.int8).reshape(-1)
                if y_full.shape[0] != n:
                    logger.warning(
                        "Skipping %s: label length mismatch (labels=%s rows=%s).",
                        sym,
                        int(y_full.shape[0]),
                        int(n),
                    )
                    continue
            y_train = y_full[:train_end]
            y_eval = y_full[eval_start:]

            if is_df or is_frame:
                train_x = _slice_rows_range(X, 0, train_end)
                eval_x = _slice_rows_range(X, eval_start, n)
                train_idx = _frame_index(train_x)
                eval_idx = _frame_index(eval_x)
                if train_idx is None:
                    train_idx = idx_ns[:train_end]
                if eval_idx is None:
                    eval_idx = idx_ns[eval_start:]
            else:
                x_np = np.asarray(X, dtype=np.float32)
                train_x = x_np[:train_end]
                eval_x = x_np[eval_start:]
                train_idx = idx_ns[:train_end]
                eval_idx = idx_ns[eval_start:]

            x_feature_names = list(d.feature_names)
            if is_df or is_frame:
                frame_names = _frame_columns(X)
                if frame_names:
                    x_feature_names = frame_names

            train_ds = PreparedDataset(
                X=train_x,
                y=y_train,
                index=train_idx,
                feature_names=list(x_feature_names),
                metadata=_slice_rows_range(meta, 0, train_end) if (_is_dataframe(meta) or _is_frame_like(meta)) else None,
                labels=y_train,
            )
            eval_ds = PreparedDataset(
                X=eval_x,
                y=y_eval,
                index=eval_idx,
                feature_names=list(x_feature_names),
                metadata=_slice_rows_range(meta, eval_start, n) if (_is_dataframe(meta) or _is_frame_like(meta)) else None,
                labels=y_eval,
            )

            train_parts.append((sym, train_ds))
            eval_map[sym] = eval_ds
            split_meta["per_symbol"][sym] = {"train_rows": int(len(train_ds.X)), "eval_rows": int(len(eval_ds.X))}

        return train_parts, eval_map, split_meta

    def _merge_symbol_shards(
        self,
        sym: str,
        sym_parts: list[PreparedDataset],
        *,
        prefer_numpy: bool = False,
    ) -> PreparedDataset | None:
        if not sym_parts:
            return None

        feature_names: list[str] = []
        seen: set[str] = set()
        prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], Any]] = []
        use_dataframe = False
        x_template: Any | None = None
        y_template: Any | None = None

        for ds in sym_parts:
            try:
                X_src = getattr(ds, "X", None)
                if X_src is None:
                    continue
                if _is_dataframe(X_src):
                    if not prefer_numpy:
                        use_dataframe = True
                        if x_template is None:
                            x_template = X_src
                    x_np = X_src.to_numpy(dtype=np.float32, copy=False)
                    idx_obj = X_src.index
                    names = [str(c) for c in list(X_src.columns)]
                elif _is_frame_like(X_src):
                    x_np, names = _frame_to_2d_float32(
                        X_src,
                        feature_names=list(getattr(ds, "feature_names", []) or []),
                    )
                    if x_np.ndim != 2:
                        continue
                    idx_obj = _frame_index(X_src)
                    if idx_obj is None:
                        idx_obj = getattr(ds, "index", None)
                else:
                    x_np = np.asarray(X_src, dtype=np.float32)
                    if x_np.ndim != 2:
                        continue
                    idx_obj = getattr(ds, "index", None)
                    names = [str(c) for c in list(getattr(ds, "feature_names", []) or [])]
                    if len(names) != x_np.shape[1]:
                        names = [f"f{i}" for i in range(x_np.shape[1])]

                rows = int(x_np.shape[0])
                if rows <= 0:
                    continue

                idx_ns = self._index_to_ns(idx_obj)
                if idx_ns is None or idx_ns.size != rows:
                    idx_ns = np.arange(rows, dtype=np.int64)
                else:
                    idx_ns = np.asarray(idx_ns, dtype=np.int64).reshape(-1)

                y_src = getattr(ds, "y", None)
                y_arr: np.ndarray | None = None
                if _is_series(y_src):
                    if y_template is None:
                        y_template = y_src
                    idx_like = None
                    if len(y_src) != rows:
                        idx_like = _frame_index(X_src)
                    y_arr = _series_like_to_int8(y_src, row_index=idx_like, n_rows=rows)
                else:
                    y_arr = np.asarray(y_src, dtype=np.int8).reshape(-1)
                if y_arr is None or y_arr.size != rows:
                    continue

                src_map = {str(name): i for i, name in enumerate(names)}
                for name in names:
                    if name not in seen:
                        seen.add(name)
                        feature_names.append(name)

                meta = getattr(ds, "metadata", None)
                if not ((_is_dataframe(meta) or _is_frame_like(meta)) and len(meta) == rows):
                    meta = None

                prepared.append(
                    (
                        np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
                        np.asarray(y_arr, dtype=np.int8).reshape(-1),
                        idx_ns,
                        src_map,
                        meta,
                    )
                )
            except Exception as exc:
                logger.warning("HPC shard parse failed for %s: %s", sym, exc)
                continue

        if not prepared:
            return None

        n_features = len(feature_names)
        feature_to_idx = {str(name): i for i, name in enumerate(feature_names)}
        x_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        idx_chunks: list[np.ndarray] = []
        meta_frames: list[Any] = []
        meta_idx_chunks: list[np.ndarray] = []

        for x_np, y_arr, idx_ns, src_map, meta in prepared:
            src_cols: list[int] = []
            dst_cols: list[int] = []
            for name, src_i in src_map.items():
                dst_i = feature_to_idx.get(name)
                if dst_i is None or src_i >= x_np.shape[1]:
                    continue
                src_cols.append(int(src_i))
                dst_cols.append(int(dst_i))
            x_aligned = _align_feature_matrix(
                x_np,
                np.asarray(src_cols, dtype=np.int64),
                np.asarray(dst_cols, dtype=np.int64),
                dst_width=n_features,
            )
            x_chunks.append(x_aligned)
            y_chunks.append(y_arr)
            idx_chunks.append(idx_ns)
            if meta is not None:
                meta_frames.append(meta)
                meta_idx_chunks.append(idx_ns)

        X_all = np.concatenate(x_chunks, axis=0)
        y_all = np.concatenate(y_chunks, axis=0).astype(np.int8, copy=False)
        idx_all = np.concatenate(idx_chunks, axis=0).astype(np.int64, copy=False)
        X_all, y_all, idx_all = _sort_dedup_rows_by_index(X_all, y_all, idx_all)

        if use_dataframe:
            idx_dt = idx_all.astype("datetime64[ns]")
            X_out: Any = _make_dataframe(X_all, columns=feature_names, index=idx_dt, template=x_template)
            y_out: Any = _make_series(y_all, index=idx_dt, dtype=np.int8, template=y_template)
            idx_out: Any = X_out.index
        else:
            X_out = X_all
            y_out = y_all
            idx_out = idx_all

        meta_out = None
        if meta_frames:
            try:
                meta_items: list[Any] = []
                for meta, meta_idx in zip(meta_frames, meta_idx_chunks, strict=False):
                    m = meta.copy()
                    if use_dataframe:
                        m.index = np.asarray(meta_idx, dtype=np.int64).astype("datetime64[ns]")
                    else:
                        m.index = np.asarray(meta_idx, dtype=np.int64)
                    meta_items.append(m)
                meta_concat = _concat_dataframes(meta_items)
                if meta_concat is None:
                    raise RuntimeError("frame concat unavailable")
                meta_out = meta_concat
                if use_dataframe:
                    meta_out = meta_out.sort_index(kind="mergesort")
                    meta_out = meta_out[~meta_out.index.duplicated(keep="first")]
                    if len(meta_out) == len(idx_dt):
                        with contextlib.suppress(Exception):
                            meta_out.index = idx_dt
                    elif len(meta_out) > len(idx_dt):
                        meta_out = _slice_rows_range(meta_out, 0, len(idx_dt))
                        with contextlib.suppress(Exception):
                            meta_out.index = idx_dt
                    else:
                        meta_out = None
                else:
                    if len(meta_out) == len(idx_all):
                        with contextlib.suppress(Exception):
                            meta_out.index = idx_all
                    elif len(meta_out) > len(idx_all):
                        meta_out = _slice_rows_range(meta_out, 0, len(idx_all))
                        with contextlib.suppress(Exception):
                            meta_out.index = idx_all
                    else:
                        meta_out = None
            except Exception:
                meta_out = None

        labels = y_out if _is_series(y_out) else np.asarray(y_out, dtype=np.int8)
        return PreparedDataset(
            X=X_out,
            y=y_out,
            index=idx_out,
            feature_names=feature_names,
            metadata=meta_out,
            labels=labels,
        )

    async def _train_global_from_datasets(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        symbols: list[str],
        optimize: bool,
        stop_event: asyncio.Event | None,
        exclude_models: list[str] | None = None,
    ) -> PreparedDataset | None:
        if not datasets:
            logger.error("Global training: no datasets provided.")
            return None
        frame_native_mode = True

        # Apply an additional cap here (covers HPC path and any callers that didn't cap during dataset creation).
        try:
            first = next((d for _sym, d in datasets if getattr(d, "X", None) is not None), None)
            n_features = int(getattr(first.X, "shape", (0, 0))[1]) if first is not None else 0
        except Exception:
            n_features = 0
        cap = self._infer_global_pool_cap_per_symbol(n_features=n_features, n_symbols=len(datasets))
        if cap is not None:
            capped: list[tuple[str, PreparedDataset]] = []
            for sym, d in datasets:
                try:
                    if len(d.X) > cap:
                        d = self._tail_dataset(d, cap)
                except Exception:
                    pass
                capped.append((sym, d))
            datasets = capped

        datasets = self._inject_cross_pair_context(datasets)

        # Align feature spaces across symbols.
        cols, aligned = self._align_global_feature_space(datasets, prefer_numpy=frame_native_mode)
        if not aligned:
            logger.error("Global training: all datasets failed to align.")
            return None

        train_ratio = float(getattr(self.settings.models, "global_train_ratio", 0.8) or 0.8)
        train_ratio = float(min(0.95, max(0.50, train_ratio)))
        embargo_bars = int(
            max(
                int(getattr(self.settings.risk, "meta_label_max_hold_bars", 0) or 0),
                int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0),
            )
        )

        train_parts, eval_map, split_meta = self._split_global_train_eval(
            aligned, train_ratio=train_ratio, embargo_bars=embargo_bars
        )
        if not train_parts:
            logger.error("Global training: no train splits produced.")
            return None
        if not eval_map:
            logger.warning("Global training: no eval splits produced; metrics will be limited.")

        # Pool training data (streaming out-of-core to avoid RAM spikes).
        pooled_meta: list[Any] = []
        for sym, d in train_parts:
            X_part = d.X
            meta_part = getattr(d, "metadata", None)
            if meta_part is None or len(meta_part) != len(X_part):
                continue
            compact_meta = _compact_ohlcv_metadata_frame(
                meta_part,
                symbol=sym if frame_native_mode else None,
            )
            if compact_meta is not None:
                pooled_meta.append(compact_meta)

        total_rows = sum(len(d.X) for _, d in train_parts)
        n_features = len(cols)
        if total_rows <= 0 or n_features <= 0:
            logger.error("Global training: pooled dataset is empty.")
            return None

        use_memmap = str(os.environ.get("FOREX_BOT_GLOBAL_POOL_MEMMAP", "1") or "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        memmap_dir: Path | None = None
        X_train: Any | None = None
        y_train: Any | None = None

        if use_memmap:
            try:
                cache_root = Path(getattr(self.settings.system, "cache_dir", "cache")) / "global_pool"
                run_id = f"{int(time.time())}_{os.getpid()}"
                memmap_dir = cache_root / f"pool_{run_id}"
                memmap_dir.mkdir(parents=True, exist_ok=True)

                (memmap_dir / "columns.json").write_text(json.dumps(cols), encoding="utf-8")
                index_kind = "datetime_ns"
                try:
                    if not all(_is_datetime_index(d.X.index) for _, d in train_parts):
                        index_kind = "none"
                except Exception:
                    index_kind = "none"
                (memmap_dir / "meta.json").write_text(
                    json.dumps({"index_kind": index_kind}),
                    encoding="utf-8",
                )

                x_path = memmap_dir / "X.npy"
                y_path = memmap_dir / "y.npy"
                idx_path = memmap_dir / "index.npy"

                logger.info(
                    f"GLOBAL: Streaming pooled dataset to memmap ({total_rows:,} rows, {n_features} features) at {memmap_dir}."
                )

                x_mm = np.lib.format.open_memmap(
                    x_path, mode="w+", dtype=np.float32, shape=(total_rows, n_features)
                )
                y_mm = np.lib.format.open_memmap(
                    y_path, mode="w+", dtype=np.int8, shape=(total_rows,)
                )
                idx_mm = None
                if index_kind != "none":
                    idx_mm = np.lib.format.open_memmap(
                        idx_path, mode="w+", dtype=np.int64, shape=(total_rows,)
                    )

                try:
                    chunk = int(os.environ.get("FOREX_BOT_MEMMAP_CHUNK_ROWS", "250000") or 250000)
                except Exception:
                    chunk = 250000
                chunk = max(10_000, min(chunk, max(10_000, total_rows)))

                offset = 0
                for _sym, d in train_parts:
                    X_src = d.X
                    if _is_dataframe(X_src):
                        x_src_np = X_src.to_numpy(dtype=np.float32, copy=False)
                    elif _is_frame_like(X_src):
                        x_src_np, _ = _frame_to_2d_float32(
                            X_src,
                            feature_names=list(getattr(d, "feature_names", []) or []),
                        )
                    else:
                        x_src_np = np.asarray(X_src, dtype=np.float32)
                    if _is_series(d.y):
                        y_src = d.y.to_numpy(dtype=np.int8, copy=False)
                    else:
                        y_src = np.asarray(d.y, dtype=np.int8).reshape(-1)
                    n = int(x_src_np.shape[0])
                    if n <= 0:
                        continue
                    if y_src.shape[0] != n:
                        raise ValueError(
                            f"Label length mismatch while pooling {_sym}: labels={y_src.shape[0]} rows={n}"
                        )
                    for start in range(0, n, chunk):
                        end = min(n, start + chunk)
                        x_mm[offset + start : offset + end] = x_src_np[start:end]
                        y_mm[offset + start : offset + end] = y_src[start:end]
                        if idx_mm is not None:
                            idx_src = _frame_index(X_src) if (_is_dataframe(X_src) or _is_frame_like(X_src)) else d.index
                            if idx_src is None:
                                idx_src = d.index
                            idx_slice = idx_src[start:end]
                            if _is_datetime_index(idx_slice):
                                idx_mm[offset + start : offset + end] = idx_slice.view("int64")
                            else:
                                idx_mm[offset + start : offset + end] = np.asarray(
                                    idx_slice, dtype=np.int64
                                )
                    offset += n

                x_mm.flush()
                y_mm.flush()
                if idx_mm is not None:
                    idx_mm.flush()

                if frame_native_mode:
                    X_train = np.load(x_path, mmap_mode="c")
                    y_train = np.load(y_path, mmap_mode="c")
                else:
                    X_mm = np.load(x_path, mmap_mode="c")
                    y_loaded = np.load(y_path, mmap_mode="c")
                    index = None
                    if index_kind != "none" and idx_path.exists():
                        try:
                            idx_ns = np.load(idx_path, mmap_mode="r")
                            index = np.asarray(idx_ns, dtype=np.int64).astype("datetime64[ns]")
                        except Exception:
                            index = None
                    X_train = _make_dataframe(X_mm, columns=cols, index=index)
                    y_train = _make_series(y_loaded, index=X_train.index, dtype=np.int8)
            except Exception as exc:
                logger.warning(
                    f"Global memmap pooling failed; falling back to in-memory: {exc}",
                    exc_info=True,
                )
                memmap_dir = None
                X_train = None
                y_train = None

        if X_train is None or y_train is None:
            logger.info(
                f"HPC: Pre-allocating master matrix for {total_rows:,} rows (in-memory fallback)."
            )
            X_train_np = np.zeros((total_rows, n_features), dtype=np.float32)
            y_train_np = np.zeros(total_rows, dtype=np.int8)

            current_offset = 0
            for _sym, d in train_parts:
                if _is_dataframe(d.X):
                    x_part = d.X.to_numpy(dtype=np.float32, copy=False)
                elif _is_frame_like(d.X):
                    x_part, _ = _frame_to_2d_float32(
                        d.X,
                        feature_names=list(getattr(d, "feature_names", []) or []),
                    )
                else:
                    x_part = np.asarray(d.X, dtype=np.float32)
                n = int(x_part.shape[0])
                X_train_np[current_offset : current_offset + n] = x_part
                if _is_series(d.y):
                    y_train_np[current_offset : current_offset + n] = d.y.to_numpy(dtype=np.int8)
                else:
                    y_arr = np.asarray(d.y, dtype=np.int8).reshape(-1)
                    y_train_np[current_offset : current_offset + n] = y_arr[:n]
                current_offset += n

            if frame_native_mode:
                X_train = X_train_np
                y_train = y_train_np
            else:
                X_train = _make_dataframe(X_train_np, columns=cols)
                y_train = _make_series(y_train_np)

            del X_train_np, y_train_np
            gc.collect()

        meta_train: Any | None = None
        if pooled_meta:
            try:
                meta_concat = _concat_dataframes(pooled_meta)
                if meta_concat is None:
                    raise RuntimeError("frame concat unavailable")
                meta_train = meta_concat
                if len(meta_train) != len(X_train):
                    logger.warning(
                        "Global training: pooled metadata row count misaligned; disabling metadata."
                    )
                    meta_train = None
                elif not frame_native_mode and _is_dataframe(X_train):
                    if not meta_train.index.equals(X_train.index):
                        logger.warning(
                            "Global training: pooled metadata index misaligned; disabling metadata for optimizer."
                        )
                        meta_train = None
                    else:
                        meta_train = meta_train.astype(
                            {"high": np.float32, "low": np.float32, "close": np.float32},
                            copy=False,
                        )
                        meta_train["symbol"] = meta_train["symbol"].astype("category")
                elif meta_train is not None and _is_frame_like(meta_train):
                    for col in ("open", "high", "low", "close"):
                        arr = _frame_column_numpy_optional(meta_train, col, dtype=np.float32)
                        if arr is not None:
                            _frame_set_column(meta_train, col, arr, dtype=np.float32)
                if meta_train is not None and (not frame_native_mode) and _is_dataframe(meta_train):
                    meta_train = meta_train.astype(
                        {"high": np.float32, "low": np.float32, "close": np.float32},
                        copy=False,
                    )
                    meta_train["symbol"] = meta_train["symbol"].astype("category")
            except Exception:
                meta_train = None

        if frame_native_mode and memmap_dir is not None and meta_train is not None:
            meta_path = memmap_dir / "metadata.pkl"
            persisted = None
            persist_fn = getattr(self.trainer, "_persist_metadata_artifact", None)
            if callable(persist_fn):
                with contextlib.suppress(Exception):
                    persisted = persist_fn(meta_train, meta_path)
            if persisted is None:
                try:
                    joblib.dump(meta_train, meta_path)
                    persisted = meta_path
                except Exception as exc:
                    logger.warning("Global training: failed to persist metadata artifact %s: %s", meta_path, exc)
            if persisted is not None:
                logger.info("Global training: persisted metadata artifact to %s", persisted)

        if frame_native_mode:
            y_arr = np.asarray(y_train, dtype=np.int8).reshape(-1)
            full_ds = PreparedDataset(
                X=np.asarray(X_train, dtype=np.float32),
                y=y_arr,
                index=np.arange(len(y_arr), dtype=np.int64),
                feature_names=list(cols),
                metadata=meta_train,
                labels=y_arr,
            )
        else:
            if _is_dataframe(X_train):
                full_ds = PreparedDataset(
                    X=X_train,
                    y=y_train,
                    index=X_train.index,
                    feature_names=list(X_train.columns),
                    # Provide symbol-aware OHLC metadata so HPO can score profitability by symbol.
                    # The trainer guards against leaking multi-symbol metadata into models that require single-series OHLC.
                    metadata=meta_train,
                    labels=y_train,
                )
            else:
                x_np = np.asarray(X_train, dtype=np.float32)
                if x_np.ndim == 1:
                    x_np = x_np.reshape(-1, 1)
                y_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int8, copy=False)
                if y_np.size != x_np.shape[0]:
                    y_np = _fit_len_array(y_np, int(x_np.shape[0]), fill=0.0, dtype=np.int8)
                idx_np = np.arange(x_np.shape[0], dtype=np.int64)
                full_ds = PreparedDataset(
                    X=x_np,
                    y=y_np,
                    index=idx_np,
                    feature_names=list(cols),
                    metadata=None,
                    labels=y_np,
                )

        # Ensure the primary symbol is stable for any symbol-dependent utilities.
        if symbols:
            self.settings.system.symbol = symbols[0]

        logger.info(
            f"GLOBAL: Training pooled dataset (symbols={len(train_parts)}, rows={len(full_ds.X):,}, "
            f"features={len(full_ds.feature_names)})"
        )
        await asyncio.to_thread(
            self.trainer.train_all,
            full_ds,
            optimize,
            stop_event,
            None,
            exclude_models,
            memmap_dataset_dir=memmap_dir,
        )

        # Post-train: evaluate each trained model out-of-sample per symbol.
        if stop_event and stop_event.is_set():
            return full_ds
        if frame_native_mode:
            self.trainer.run_summary["global_training"] = {
                "symbols": list(symbols),
                "train_ratio": float(train_ratio),
                "feature_columns": list(cols),
                "frame_native": True,
                **split_meta,
            }
            with contextlib.suppress(Exception):
                self.trainer.persistence.save_run_summary(self.trainer.run_summary)
            logger.info("Global post-train evaluation skipped in frame-native mode.")
            return full_ds

        try:
            import inspect

            from ..strategy.fast_backtest import (
                fast_evaluate_strategy,
                infer_pip_metrics,
                infer_sl_tp_pips_auto,
            )
            from ..training.evaluation import probs_to_signals, prop_backtest

            model_metrics: dict[str, Any] = {}
            col_to_idx_eval = {str(col): i for i, col in enumerate(cols)}
            for name, model in (self.trainer.models or {}).items():
                per_symbol: dict[str, Any] = {}
                for sym, ds_eval in eval_map.items():
                    X_eval_src = ds_eval.X
                    if _is_dataframe(X_eval_src):
                        x_src = X_eval_src.to_numpy(dtype=np.float32, copy=False)
                        src_names = [str(c) for c in list(X_eval_src.columns)]
                        src_cols, dst_cols = _column_index_mapping(src_names, col_to_idx_eval)
                        X_np = _align_feature_matrix(
                            x_src,
                            src_cols,
                            dst_cols,
                            dst_width=len(cols),
                        )
                        X_e = _make_dataframe(X_np, columns=cols, index=X_eval_src.index)
                    else:
                        x_np, src_names = _frame_to_2d_float32(
                            X_eval_src,
                            feature_names=list(getattr(ds_eval, "feature_names", []) or []),
                        )
                        src_cols, dst_cols = _column_index_mapping(
                            [str(col) for col in src_names],
                            col_to_idx_eval,
                        )
                        X_e = _align_feature_matrix(
                            x_np,
                            src_cols,
                            dst_cols,
                            dst_width=len(cols),
                        )
                    meta_e = ds_eval.metadata
                    if meta_e is None or not (_is_dataframe(meta_e) or _is_frame_like(meta_e)):
                        continue
                    meta_rows = int(_frame_len(meta_e))
                    if meta_rows != len(X_e):
                        continue
                    close_arr = _frame_column_numpy_optional(meta_e, "close", dtype=np.float64)
                    high_arr = _frame_column_numpy_optional(meta_e, "high", dtype=np.float64)
                    low_arr = _frame_column_numpy_optional(meta_e, "low", dtype=np.float64)
                    if close_arr is None or high_arr is None or low_arr is None:
                        continue

                    pred_kwargs: dict[str, Any] = {}
                    try:
                        sig = inspect.signature(model.predict_proba)
                        if "metadata" in sig.parameters:
                            pred_kwargs["metadata"] = meta_e
                    except Exception:
                        pass

                    try:
                        probs = model.predict_proba(X_e, **pred_kwargs)
                        sig_arr = np.asarray(probs_to_signals(np.asarray(probs)), dtype=np.int8).reshape(-1)
                        n_eval = int(min(len(sig_arr), close_arr.size, high_arr.size, low_arr.size, meta_rows))
                        if n_eval <= 1:
                            continue
                        sig_eval = sig_arr[:n_eval]
                        close_eval = np.asarray(close_arr[:n_eval], dtype=np.float64)
                        high_eval = np.asarray(high_arr[:n_eval], dtype=np.float64)
                        low_eval = np.asarray(low_arr[:n_eval], dtype=np.float64)
                        open_arr = _frame_column_numpy_optional(meta_e, "open", dtype=np.float64)
                        open_eval = np.asarray(open_arr[:n_eval], dtype=np.float64) if open_arr is not None else close_eval
                        atr_arr = _frame_column_numpy_optional(meta_e, "atr", dtype=np.float64)
                        atr_eval = np.asarray(atr_arr[:n_eval], dtype=np.float64) if atr_arr is not None else None
                        idx_eval = _frame_index(meta_e)
                        idx_ns = self._index_to_ns(idx_eval)
                        if idx_ns is None or idx_ns.size < n_eval:
                            idx_ns = np.arange(n_eval, dtype=np.int64)
                        else:
                            idx_ns = np.asarray(idx_ns, dtype=np.int64).reshape(-1)[:n_eval]
                        meta_payload: dict[str, Any] = {
                            "open": open_eval,
                            "high": high_eval,
                            "low": low_eval,
                            "close": close_eval,
                            "index": idx_ns,
                            "symbol": sym,
                        }
                        if atr_eval is not None:
                            meta_payload["atr"] = atr_eval

                        prop = prop_backtest(
                            meta_payload,
                            sig_eval,
                            max_daily_dd_pct=float(getattr(self.settings.risk, "daily_drawdown_limit", 0.05)),
                            daily_dd_warn_pct=float(getattr(self.settings.risk, "daily_drawdown_limit", 0.05)) * 0.8,
                            max_trades_per_day=int(getattr(self.settings.risk, "max_trades_per_day", 10)),
                            use_gpu=False,
                        )

                        month_idx, day_idx = self._month_day_indices_from_index(idx_ns)
                        if month_idx.size != n_eval or day_idx.size != n_eval:
                            continue

                        pip_size, pip_value_per_lot = infer_pip_metrics(sym)
                        sl_cfg = getattr(self.settings.risk, "meta_label_sl_pips", None)
                        tp_cfg = getattr(self.settings.risk, "meta_label_tp_pips", None)
                        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
                        if sl_cfg is None or float(sl_cfg) <= 0:
                            auto = infer_sl_tp_pips_auto(
                                open_prices=open_eval,
                                high_prices=high_eval,
                                low_prices=low_eval,
                                close_prices=close_eval,
                                atr_values=atr_eval,
                                pip_size=pip_size,
                                atr_mult=float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5)),
                                min_rr=rr,
                                min_dist=float(getattr(self.settings.risk, "meta_label_min_dist", 0.0)),
                                settings=self.settings,
                            )
                            if auto is None:
                                raise RuntimeError("Cannot infer SL/TP pips from metadata.")
                            sl_pips, tp_pips = auto
                        else:
                            sl_pips = float(sl_cfg)
                            if tp_cfg is None or float(tp_cfg) <= 0:
                                tp_pips = sl_pips * rr
                            else:
                                tp_pips = max(float(tp_cfg), sl_pips * rr)

                        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
                        commission = float(getattr(self.settings.risk, "commission_per_lot", 0.0))
                        max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
                        trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
                        trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
                        trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)

                        arr = fast_evaluate_strategy(
                            close_prices=close_eval,
                            high_prices=high_eval,
                            low_prices=low_eval,
                            signals=sig_eval,
                            month_indices=month_idx,
                            day_indices=day_idx,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
                            max_hold_bars=max_hold,
                            trailing_enabled=trailing_enabled,
                            trailing_atr_multiplier=trailing_mult,
                            trailing_be_trigger_r=trailing_trigger_r,
                            pip_value=pip_size,
                            spread_pips=spread,
                            commission_per_trade=commission,
                            pip_value_per_lot=pip_value_per_lot,
                        )
                        keys = [
                            "net_profit",
                            "sharpe",
                            "sortino",
                            "max_dd",
                            "win_rate",
                            "profit_factor",
                            "expectancy",
                            "sqn",
                            "trades",
                            "consistency_score",
                            "daily_dd",
                        ]
                        fast = {k: float(v) for k, v in zip(keys, arr.tolist(), strict=False)}

                        per_symbol[sym] = {"prop": prop, "fast": fast}
                    except Exception as exc:
                        logger.debug(f"Global eval failed for {name} on {sym}: {exc}")
                        continue

                prop_list = [v.get("prop", {}) for v in per_symbol.values() if isinstance(v, dict)]
                fast_list = [v.get("fast", {}) for v in per_symbol.values() if isinstance(v, dict)]

                agg_prop = self._aggregate_metrics(
                    prop_list,
                    numeric_keys=["pnl_score", "win_rate", "max_dd_pct", "trades"],
                    bool_any_keys=["daily_dd_violation", "trade_limit_violation"],
                )
                agg_fast = self._aggregate_metrics(
                    fast_list,
                    numeric_keys=[
                        "net_profit",
                        "sharpe",
                        "sortino",
                        "max_dd",
                        "win_rate",
                        "profit_factor",
                        "expectancy",
                        "sqn",
                        "trades",
                        "consistency_score",
                        "daily_dd",
                    ],
                )

                model_metrics[name] = {"prop": agg_prop, "fast": agg_fast, "per_symbol": per_symbol}

            self.trainer.run_summary["global_training"] = {
                "symbols": list(symbols),
                "train_ratio": float(train_ratio),
                "feature_columns": list(cols),
                **split_meta,
            }
            if model_metrics:
                self.trainer.run_summary["model_metrics"] = model_metrics
            self.trainer.persistence.save_run_summary(self.trainer.run_summary)
        except Exception as exc:
            logger.warning(f"Global post-train evaluation skipped: {exc}", exc_info=True)

        return full_ds

    async def _train_global_hpc(self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None) -> None:
        logger.info("?? HPC Mode Active: Switching to Parallel Global Training.")

        # Single-process mode for stability (Avoids NCCL timeouts)
        # Cache path for data persistence
        cache_path = Path(self.settings.system.cache_dir) / "hpc_datasets.pkl"
        datasets: list[tuple[str, PreparedDataset]] = []
        datasets_loaded = False

        # --- Data Loading & Feature Engineering ---
        reuse_cache = str(os.environ.get("FOREX_BOT_HPC_DATASET_CACHE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        if reuse_cache and cache_path.exists():
            try:
                cached = joblib.load(cache_path)
                if isinstance(cached, list) and cached:
                    datasets = cached
                    datasets_loaded = True
                    logger.info(f"HPC: Loaded cached datasets from {cache_path}.")
            except Exception as e:
                logger.warning(f"HPC: Failed to load cached datasets: {e}")

        if not datasets_loaded:
            raw_frames_map = {}
            news_map: dict[str, Any | None] = {}
            
            analyzer = (
                await get_sentiment_analyzer(self.settings)
                if self.settings.news.enable_news and not self._rust_only_enabled()
                else None
            )

            # 1. Parallel loading of raw data (using asyncio.gather for speed)
            logger.info(f"HPC: Loading raw data for {len(symbols)} symbols in parallel...")
            
            async def _load_single(s):
                await self.data_loader.ensure_history(s)
                f = await self.data_loader.get_training_data(s)
                n = self._build_news_features(analyzer, s, f) if analyzer else None
                return s, f, n

            load_results = await asyncio.gather(*[_load_single(s) for s in symbols])
            for sym, f, n in load_results:
                raw_frames_map[sym] = f
                news_map[sym] = n
                logger.info(f"HPC: Ready data for {sym}")

            # 2. Hyper-Parallel Feature Engineering (ZERO-COPY HPC)
            cpu_total = max(1, os.cpu_count() or 1)
            cpu_reserve = self._parse_int_env("FOREX_BOT_CPU_RESERVE")
            if cpu_reserve is None:
                cpu_reserve = 1
            feature_cpu_env = self._parse_int_env("FOREX_BOT_FEATURE_CPU_BUDGET")
            cpu_budget_env = feature_cpu_env if feature_cpu_env is not None else self._parse_int_env("FOREX_BOT_CPU_BUDGET")
            if cpu_budget_env is not None and cpu_budget_env > 0:
                cpu_budget = max(1, min(cpu_total, cpu_budget_env))
            else:
                cpu_budget = max(1, cpu_total - max(0, cpu_reserve))

            per_worker_gb = 6.0
            try:
                per_worker_gb = float(os.environ.get("FOREX_BOT_FEATURE_WORKER_GB", "6.0") or 6.0)
            except Exception:
                per_worker_gb = 6.0

            available_gb = None
            with contextlib.suppress(Exception):
                import psutil

                available_gb = float(psutil.virtual_memory().available) / (1024**3)
            if available_gb is None:
                try:
                    available_gb = float(os.environ.get("FOREX_BOT_RAM_GB", 0) or 0)
                except Exception:
                    available_gb = 0.0
            if available_gb <= 0:
                available_gb = 16.0

            max_ram_workers = int(available_gb // max(0.5, per_worker_gb))
            requested_workers = self._parse_int_env("FOREX_BOT_FEATURE_WORKERS")
            if requested_workers is not None and requested_workers > 0:
                max_workers = max(1, min(cpu_budget, requested_workers))
                recommended = max(1, min(cpu_budget, max_ram_workers))
                if max_workers > recommended:
                    logger.warning(
                        "HPC: Requested %s feature workers exceeds RAM-safe "
                        "recommendation %s (available_ram=%.1fGB, per_worker_gb=%.2f).",
                        max_workers,
                        recommended,
                        available_gb,
                        per_worker_gb,
                    )
            else:
                max_workers = max(1, min(cpu_budget, max_ram_workers))

            # Cap runaway worker counts to avoid process storms on large boxes.
            max_shards_per_symbol = self._parse_int_env("FOREX_BOT_HPC_SHARDS_PER_SYMBOL")
            if max_shards_per_symbol is None or max_shards_per_symbol <= 0:
                max_shards_per_symbol = 8
            max_workers_cap = max(1, min(cpu_budget, len(symbols) * max_shards_per_symbol))
            if max_workers > max_workers_cap:
                logger.warning(
                    "HPC: Capping feature workers to %s (symbols=%s, max_shards_per_symbol=%s).",
                    max_workers_cap,
                    len(symbols),
                    max_shards_per_symbol,
                )
                max_workers = max_workers_cap

            feature_threads = self._parse_int_env("FOREX_BOT_FEATURE_WORKER_THREADS")
            if feature_threads is not None and feature_threads > 0:
                max_workers = max(1, min(max_workers, max(1, cpu_budget // feature_threads)))
                worker_threads = max(1, feature_threads)
            else:
                worker_threads = max(1, cpu_budget // max_workers)

            os.environ.setdefault("FOREX_BOT_FEATURE_WORKERS", str(max_workers))
            logger.info(
                "HPC: Initializing Zero-Copy Sharding for %s workers "
                "(cpu_budget=%s, threads/worker=%s, available_ram=%.1fGB).",
                max_workers,
                cpu_budget,
                worker_threads,
                available_gb,
            )
            
            # Implementation Note: We use a simplified sharding for now to avoid complexity,
            # but we force the workers to use the shared RAM space.
            gpu_count = 0
            force_cpu = str(os.environ.get("FOREX_BOT_FEATURE_CPU_ONLY", "1")).strip().lower() in {
                "1", "true", "yes", "on"
            }
            if not force_cpu:
                with contextlib.suppress(Exception):
                    import torch

                    if torch.cuda.is_available():
                        gpu_count = int(torch.cuda.device_count())

            all_tasks = []
            shards_by_symbol: dict[str, int] = {}
            worker_mode = str(os.environ.get("FOREX_BOT_HPC_WORKER_MODE", "shard")).strip().lower()
            if worker_mode not in {"shard", "symbol"}:
                worker_mode = "shard"
            if worker_mode == "symbol":
                max_workers = max(1, min(max_workers, len(symbols)))
            for sym, frames in raw_frames_map.items():
                base_df = frames.get(self.settings.system.base_timeframe)
                if base_df is None or (_is_dataframe(base_df) and base_df.empty):
                    base_df = frames.get("M1")
                if base_df is None or (_is_dataframe(base_df) and base_df.empty):
                    continue
                
                # Shard each symbol into time-slices (or run single shard per symbol)
                if worker_mode == "symbol":
                    n_shards = 1
                else:
                    n_shards = max(1, max_workers // len(symbols))
                shards_by_symbol[sym] = n_shards
                if n_shards == 1:
                    chunk_indices = [np.arange(len(base_df))]
                else:
                    chunk_indices = np.array_split(np.arange(len(base_df)), n_shards)
                
                for i, idx_range in enumerate(chunk_indices):
                    if len(idx_range) == 0: continue
                    # Extract slice
                    s = int(idx_range[0])
                    e = int(idx_range[-1]) + 1
                    chunk_frames = {tf: _slice_rows_range(df, s, e) for tf, df in frames.items()}
                    assigned_gpu = (len(all_tasks) % gpu_count) if gpu_count > 0 else 0
                    all_tasks.append({
                        "sym": sym,
                        "frames": chunk_frames,
                        "shard_id": i,
                        "gpu": assigned_gpu
                    })

            logger.info("HPC: Dispatching %s zero-copy tasks (mode=%s).", len(all_tasks), worker_mode)
            if all_tasks and max_workers > len(all_tasks):
                max_workers = len(all_tasks)
                logger.info("HPC: Reducing worker pool to %s (task count bound).", max_workers)
            datasets_parts = []
            ctx_name = "spawn"
            if force_cpu:
                try:
                    import sys

                    if sys.platform != "win32":
                        ctx_name = "fork"
                except Exception:
                    pass
            spawn_ctx = multiprocessing.get_context(ctx_name)
            logger.info(
                "HPC: Using %s start method for feature workers (force_cpu=%s).",
                ctx_name,
                force_cpu,
            )
            
            # Use a smaller chunksize to prevent the 'Pickle Stalling'
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
                futures = {
                    executor.submit(
                        _hpc_feature_worker,
                        self.settings.model_copy(),
                        t["frames"],
                        t["sym"],
                        news_map.get(t["sym"]),
                        t["gpu"],
                        worker_threads,
                    ): t for t in all_tasks
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                        if res: datasets_parts.append(res)
                    except Exception as e:
                        task = futures.get(fut, {})
                        logger.error(
                            "HPC shard failed (symbol=%s, shard=%s): %s",
                            task.get("sym"),
                            task.get("shard_id"),
                            e,
                            exc_info=True,
                        )

            # HPC FIX: Fault-Tolerant Re-assembly (Resilience against worker crashes)
            logger.info("HPC: Consolidating successful data shards...")
            for sym in symbols:
                # Filter shards for this symbol that returned valid data
                sym_parts: list[PreparedDataset] = []
                for p in datasets_parts:
                    ds = None
                    p_sym = ""
                    if isinstance(p, dict):
                        p_sym = str(p.get("symbol") or "")
                        ds = p.get("dataset")
                    elif isinstance(p, (tuple, list)) and len(p) >= 2:
                        p_sym = str(p[0] or "")
                        ds = p[1]
                    else:
                        p_sym = str(getattr(p, "symbol", "") or "")
                        ds = p
                    if p_sym == sym and ds is not None and getattr(ds, "X", None) is not None:
                        sym_parts.append(ds)
                
                if not sym_parts: 
                    logger.error(f"HPC FAILURE: All shards for {sym} failed! Skipping symbol.")
                    continue
                
                expected_shards = shards_by_symbol.get(sym, n_shards)
                if len(sym_parts) < expected_shards:
                    logger.warning(
                        f"HPC WARNING: Only {len(sym_parts)}/{expected_shards} shards succeeded for {sym}. "
                        "Proceeding with partial data."
                    )
                
                # Re-assemble using only valid parts
                try:
                    full_ds = self._merge_symbol_shards(sym, sym_parts, prefer_numpy=True)
                    if full_ds is None or getattr(full_ds, "X", None) is None or len(full_ds.X) <= 0:
                        raise RuntimeError("empty merged shard dataset")
                    datasets.append((sym, full_ds))
                    logger.info(
                        "HPC: Successfully recovered %s rows for %s (features=%s).",
                        len(full_ds.X),
                        sym,
                        len(getattr(full_ds, "feature_names", []) or []),
                    )
                except Exception as merge_err:
                    logger.error(f"HPC ERROR: Failed to merge shards for {sym}: {merge_err}")

            # Save cache for next time
            if reuse_cache and datasets:
                try:
                    joblib.dump(datasets, cache_path)
                except Exception as e:
                    logger.warning(f"HPC: Failed to save dataset cache: {e}")

        if not datasets:
            logger.error("HPC FATAL: No datasets were successfully prepared. Cannot proceed with training.")
            logger.warning("HPC fallback: switching to sequential global training.")
            await self._train_global_sequential(symbols, optimize, stop_event)
            return

        # --- Launch Discovery (Uses internal 8-GPU ThreadPool) ---
        has_gpu = bool(getattr(self.settings.system, "enable_gpu", False)) and int(
            getattr(self.settings.system, "num_gpus", 0) or 0
        ) > 0
        if has_gpu:
            logger.info("Launching GPU-Native Expert Discovery (Multi-GPU Pool)...")
        else:
            logger.info("Launching CPU-Native Expert Discovery (Multi-worker Pool)...")
        from ..strategy.discovery_tensor import TensorDiscoveryEngine
        
        target_sym = "EURUSD" if "EURUSD" in [s for s, _ in datasets] else datasets[0][0]
        target_ds = next(ds for sym, ds in datasets if sym == target_sym)
        
        # Load raw frame for OHLC data
        await self.data_loader.ensure_history(target_sym)
        raw_frames = await self.data_loader.get_training_data(target_sym)

        discovery_frames, timeframes = self._build_discovery_frames_for_tensor(
            raw_frames, None, target_sym, base_dataset=target_ds
        )

        if bool(getattr(self.settings.models, "prop_search_enabled", False)):
            if self._prop_search_async_enabled():
                logger.info(
                    "[STRATEGY DISCOVERY] Async prop-search requested but running inline so discovered signals feed model training."
                )
            await self._run_prop_search_for_symbols(symbols, stop_event=stop_event)
            patched_hpc: list[tuple[str, PreparedDataset]] = []
            for sym, ds in datasets:
                patched_hpc.append(
                    (
                        sym,
                        self._apply_prop_discovered_base_signal(
                            ds,
                            symbol=sym,
                            source_df=ds.metadata,
                        ),
                    )
                )
            datasets = patched_hpc

        def _run_discovery() -> None:
            has_gpu_local = bool(getattr(self.settings.system, "enable_gpu", False)) and int(
                getattr(self.settings.system, "num_gpus", 0) or 0
            ) > 0
            discovery_experts = self._parse_int_env("FOREX_BOT_DISCOVERY_EXPERTS") or 100
            discovery_iterations = self._parse_int_env("FOREX_BOT_DISCOVERY_ITERS") or 1000
            discovery_experts, discovery_iterations = self._normalize_discovery_budget(
                experts=discovery_experts,
                iterations=discovery_iterations,
                has_gpu=has_gpu_local,
            )

            logger.info(
                "[STRATEGY DISCOVERY] mode=%s experts=%s iterations=%s",
                "gpu" if has_gpu_local else "cpu",
                discovery_experts,
                discovery_iterations,
            )
            discovery_tensor = TensorDiscoveryEngine(
                device="cuda" if has_gpu_local else "cpu",
                n_experts=discovery_experts,
                timeframes=timeframes,
                settings=self.settings,
            )
            discovery_tensor.run_unsupervised_search(discovery_frames, iterations=discovery_iterations)
            discovery_tensor.save_experts(self.settings.system.cache_dir + "/tensor_knowledge.pt")

        if self._discovery_async_enabled():
            logger.info(
                "[STRATEGY DISCOVERY] Async mode requested but running inline so discovery feeds models before training."
            )
        _run_discovery()

        # --- Final Global Training ---
        full_ds = await self._train_global_from_datasets(
            datasets, symbols, optimize, stop_event, exclude_models=None
        )
        # If discovery/prop search is async, optionally wait for it after training completes.
        if self._prop_search_thread and self._prop_search_thread.is_alive():
            if self._prop_search_async_wait() and not (stop_event and stop_event.is_set()):
                logger.info(
                    "[STRATEGY DISCOVERY] Waiting for background discovery to finish "
                    "(set FOREX_BOT_PROP_SEARCH_ASYNC_WAIT=0 to skip)."
                )
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._prop_search_thread.join)
        if self._discovery_thread and self._discovery_thread.is_alive():
            if self._prop_search_async_wait() and not (stop_event and stop_event.is_set()):
                logger.info(
                    "[STRATEGY DISCOVERY] Waiting for background tensor discovery to finish "
                    "(set FOREX_BOT_DISCOVERY_ASYNC=0 to run inline)."
                )
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._discovery_thread.join)

    async def _train_global_sequential(
        self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None
    ) -> None:
        logger.info(
            "GLOBAL: Building pooled dataset sequentially (feature engineering per symbol, then one pooled train)."
        )
        datasets: list[tuple[str, PreparedDataset]] = []
        total = len(symbols)

        analyzer = None
        if self.settings.news.enable_news and not self._rust_only_enabled():
            try:
                analyzer = await get_sentiment_analyzer(self.settings)
            except Exception as exc:
                logger.warning(f"News analyzer unavailable (global): {exc}")

        for idx, sym in enumerate(symbols, start=1):
            if stop_event and stop_event.is_set():
                break

            logger.info(f"[GLOBAL {idx}/{total}] Preparing features for {sym}...")
            try:
                self.settings.system.symbol = sym
                has_data = await self.data_loader.ensure_history(sym)
                if not isinstance(has_data, bool) or not has_data:
                    continue

                frames = await self.data_loader.get_training_data(sym)
                if not isinstance(frames, dict) or not frames:
                    continue

                news_feats = self._build_news_features(analyzer, sym, frames) if analyzer is not None else None
                ds = self.feature_engineer.prepare(frames, news_features=news_feats, symbol=sym)

                cap = self._infer_global_pool_cap_per_symbol(n_features=int(ds.X.shape[1]), n_symbols=len(symbols))
                if cap is not None and len(ds.X) > cap:
                    logger.info(
                        f"[GLOBAL {idx}/{total}] Capping {sym} dataset from {len(ds.X):,} -> {cap:,} rows "
                        "(override via FOREX_BOT_GLOBAL_MAX_ROWS[_PER_SYMBOL])."
                    )
                    ds = self._tail_dataset(ds, cap)
                datasets.append((sym, ds))
            except Exception as e:
                logger.error(f"Failed to prepare {sym}: {e}", exc_info=True)
                continue

        # Ensure prop-search runs first and discovered strategies feed model training.
        if datasets and bool(getattr(self.settings.models, "prop_search_enabled", False)):
            symbols_to_run = [sym for sym, _ in datasets]
            if self._prop_search_async_enabled():
                logger.info(
                    "[STRATEGY DISCOVERY] Async prop-search requested but running inline so discovered signals feed model training."
                )
            await self._run_prop_search_for_symbols(symbols_to_run, stop_event=stop_event)
            patched: list[tuple[str, PreparedDataset]] = []
            for sym, ds in datasets:
                patched.append(
                    (
                        sym,
                        self._apply_prop_discovered_base_signal(
                            ds,
                            symbol=sym,
                            source_df=ds.metadata,
                        ),
                    )
                )
            datasets = patched

        # Ensure tensor strategy discovery runs before pooled model training.
        if datasets:
            try:
                has_gpu_local = bool(getattr(self.settings.system, "enable_gpu", False)) and int(
                    getattr(self.settings.system, "num_gpus", 0) or 0
                ) > 0
                if has_gpu_local:
                    logger.info("Launching GPU-Native Expert Discovery (Sequential Global Path)...")
                else:
                    logger.info("Launching CPU-Native Expert Discovery (Sequential Global Path)...")

                from ..strategy.discovery_tensor import TensorDiscoveryEngine

                target_sym = "EURUSD" if "EURUSD" in [s for s, _ in datasets] else datasets[0][0]
                target_ds = next(ds for sym, ds in datasets if sym == target_sym)

                await self.data_loader.ensure_history(target_sym)
                raw_frames = await self.data_loader.get_training_data(target_sym)
                if isinstance(raw_frames, dict) and raw_frames:
                    news_feats = self._build_news_features(analyzer, target_sym, raw_frames) if analyzer else None
                    discovery_frames, timeframes = self._build_discovery_frames_for_tensor(
                        raw_frames,
                        news_feats,
                        target_sym,
                        base_dataset=target_ds,
                    )

                    discovery_experts = self._parse_int_env("FOREX_BOT_DISCOVERY_EXPERTS") or 100
                    discovery_iterations = self._parse_int_env("FOREX_BOT_DISCOVERY_ITERS") or 1000
                    discovery_experts, discovery_iterations = self._normalize_discovery_budget(
                        experts=discovery_experts,
                        iterations=discovery_iterations,
                        has_gpu=has_gpu_local,
                    )
                    logger.info(
                        "[STRATEGY DISCOVERY] mode=%s experts=%s iterations=%s",
                        "gpu" if has_gpu_local else "cpu",
                        discovery_experts,
                        discovery_iterations,
                    )

                    discovery_tensor = TensorDiscoveryEngine(
                        device="cuda" if has_gpu_local else "cpu",
                        n_experts=discovery_experts,
                        timeframes=timeframes,
                        max_rows=int(getattr(self.settings.system, "discovery_max_rows", 0) or 0),
                        stream_mode=bool(getattr(self.settings.system, "discovery_stream", False)),
                        auto_cap=bool(getattr(self.settings.system, "discovery_auto_cap", True)),
                        settings=self.settings,
                    )
                    discovery_tensor.run_unsupervised_search(
                        discovery_frames,
                        news_features=news_feats,
                        iterations=discovery_iterations,
                    )
                    discovery_tensor.save_experts(self.settings.system.cache_dir + "/tensor_knowledge.pt")
                else:
                    logger.warning("[STRATEGY DISCOVERY] No raw frames available for sequential global path.")
            except Exception as exc:
                logger.warning(
                    f"[STRATEGY DISCOVERY] Sequential global tensor discovery failed: {exc}",
                    exc_info=True,
                )

        await self._train_global_from_datasets(datasets, symbols, optimize, stop_event)

        # If discovery is async, optionally wait for it to finish after training.
        if self._prop_search_thread and self._prop_search_thread.is_alive():
            if self._prop_search_async_wait() and not (stop_event and stop_event.is_set()):
                logger.info(
                    "[STRATEGY DISCOVERY] Waiting for background discovery to finish "
                    "(set FOREX_BOT_PROP_SEARCH_ASYNC_WAIT=0 to skip)."
                )
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._prop_search_thread.join)

    def _maybe_start_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE, _maybe_init_ray

        try:
            # Always attempt to start Ray when available; ignore feature flags.
            if RAY_AVAILABLE and not self._ray_started:
                if _maybe_init_ray():
                    self._ray_started = True
                    logger.info("Ray initialized for RLlib agents.")
        except Exception as exc:
            logger.warning(f"Ray init skipped: {exc}")

    def _maybe_stop_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE

        try:
            if self._ray_started and RAY_AVAILABLE:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shutdown.")
        except Exception as e:
            logger.warning(f"Ray shutdown failed: {e}", exc_info=True)


# Standalone worker function for ProcessPoolExecutor (must be picklable)
def _hpc_feature_worker(settings, frames, sym, news_features=None, assigned_gpu=0, worker_threads=1):
    try:
        import sys
        import os
        from pathlib import Path

        # Limit internal math threads to 1 per worker to prevent thrashing
        threads = max(1, int(worker_threads or 1))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
        os.environ["FOREX_BOT_CPU_BUDGET"] = str(threads)
        os.environ["FOREX_BOT_CPU_THREADS"] = str(threads)

        # Optionally force CPU-only feature engineering to avoid GPU OOM in workers.
        force_cpu = str(os.environ.get("FOREX_BOT_FEATURE_CPU_ONLY", "1")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                settings.system.enable_gpu_preference = "cpu"
            except Exception:
                pass
        else:
            # Set GPU affinity before any torch/cupy imports
            os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        
        # Add src/ to path explicitly to ensure latest code is loaded in the worker process
        project_root = Path(__file__).resolve().parents[3]
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Re-instantiate FE inside worker
        from forex_bot.features.pipeline import FeatureEngineer

        fe = FeatureEngineer(settings)
        ds = fe.prepare(frames, news_features=news_features, symbol=sym)
        return {"symbol": sym, "dataset": ds}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Worker failed for {sym}: {e}", exc_info=True)
        return None


