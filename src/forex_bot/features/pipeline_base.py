from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class LabelConfig:
    horizon: int
    min_dist: float
    use_triple_barrier: bool
    max_hold: int
    sl_pips: float | None
    tp_pips: float | None

class NumpyFrame:
    """Minimal frame-like metadata container for frame-native pipelines."""
    def __init__(self, data: dict[str, Any], *, index: Any | None = None, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        n_rows = 0
        if self._data:
            try:
                n_rows = int(next(iter(self._data.values())).shape[0])
            except Exception:
                n_rows = 0
        if index is None:
            self.index = np.arange(n_rows, dtype=np.int64)
        else:
            idx = np.asarray(index).reshape(-1)
            if idx.size == n_rows:
                self.index = idx
            elif idx.size <= 0:
                self.index = np.arange(n_rows, dtype=np.int64)
            elif idx.size > n_rows:
                self.index = idx[:n_rows]
            else:
                pad = np.full(n_rows - idx.size, idx[-1], dtype=idx.dtype)
                self.index = np.concatenate([idx, pad])
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self, deep: bool = False) -> "NumpyFrame":
        return NumpyFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            index=np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )

    def to_numpy(self, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if not self.columns:
            arr = np.zeros((len(self), 0), dtype=np.float32)
        else:
            arr = np.column_stack([np.asarray(self._data[col]).reshape(-1) for col in self.columns])
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return np.array(arr, copy=True) if copy else np.asarray(arr)

def tf_minutes(tf: str) -> int:
    return {
        "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200,
    }.get(str(tf or "").upper(), 10**9)

def index_to_ns_like(index: Any) -> np.ndarray | None:
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
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    if arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
        vmax = int(np.max(np.abs(vals))) if vals.size > 0 else 0
        if vmax > 10**14: return vals
        if vmax > 10**11: return vals * 1_000_000
        return vals * 1_000_000_000
    return None

def align_series_by_ts(target_ts: np.ndarray, source_ts: np.ndarray, values: Any, *, default: float = 0.0, dtype: Any = np.float32) -> np.ndarray:
    tgt = np.asarray(target_ts, dtype=np.int64).reshape(-1)
    src = np.asarray(source_ts, dtype=np.int64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if tgt.size == 0: return np.zeros(0, dtype=dtype)
    if vals.size == 0 or src.size == 0: return np.full(tgt.shape, float(default), dtype=dtype)
    
    # Simple searchsorted matching for alignment
    pos = np.searchsorted(src, tgt, side="right") - 1
    out = np.full(tgt.shape, float(default), dtype=np.float64)
    valid = pos >= 0
    if np.any(valid):
        out[valid] = vals[np.clip(pos[valid], 0, vals.size - 1)]
    return out.astype(dtype, copy=False)
