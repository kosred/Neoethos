import contextlib
import numpy as np
from typing import Any

def fit_len_array(values: Any, n: int, *, fill: float = 0.0, dtype: Any = np.float32) -> np.ndarray:
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

class NumpyFrame:
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
            vals = fit_len_array(vals, n, fill=0.0, dtype=vals.dtype if vals.size > 0 else np.float32)
        self._data[name] = vals
        if name not in self.columns:
            self.columns.append(name)

    def copy(self, deep: bool = False) -> 'NumpyFrame':
        _ = deep
        out = NumpyFrame(
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

    def tail(self, n: int) -> 'NumpyFrame':
        take = max(0, int(n))
        if take <= 0:
            return NumpyFrame(
                {k: v[:0] for k, v in self._data.items()},
                index=self.index[:0],
                attrs=dict(self.attrs),
            )
        return NumpyFrame(
            {k: v[-take:] for k, v in self._data.items()},
            index=self.index[-take:],
            attrs=dict(self.attrs),
        )

def is_dataframe(value: Any) -> bool:
    return bool(
        hasattr(value, "columns")
        and hasattr(value, "index")
        and callable(getattr(value, "to_numpy", None))
    )

def is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))

def is_series(value: Any) -> bool:
    return bool(hasattr(value, "index") and hasattr(value, "to_numpy") and not hasattr(value, "columns"))

def slice_rows_range(obj: Any, start: int, end: int) -> Any:
    if obj is None:
        return None
    if start >= end:
        if is_dataframe(obj): return obj.iloc[0:0]
        if isinstance(obj, NumpyFrame): 
            return NumpyFrame({k: v[:0] for k, v in obj._data.items()}, index=obj.index[:0], attrs=dict(obj.attrs))
        if hasattr(obj, "shape") or isinstance(obj, np.ndarray): return obj[:0]
        return obj
    if hasattr(obj, "iloc"):
        return obj.iloc[start:end]
    if isinstance(obj, NumpyFrame):
        return NumpyFrame(
            {k: v[start:end] for k, v in obj._data.items()},
            index=obj.index[start:end],
            attrs=dict(obj.attrs)
        )
    try:
        return obj[start:end]
    except Exception:
        return obj

def is_datetime_index(value: Any) -> bool:
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

def index_to_ns_int64(index: Any) -> np.ndarray | None:
    if index is None:
        return None
    try:
        if hasattr(index, "asi8"):
            return np.asarray(index.asi8, dtype=np.int64).reshape(-1)
    except Exception:
        pass
    arr = np.asarray(index).reshape(-1)
    if arr.size <= 0:
        return None
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)

def to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        with contextlib.suppress(Exception):
            out = values.to_numpy(dtype=dtype, copy=False)
            return np.asarray(out, dtype=dtype).reshape(-1)
    return np.asarray(values, dtype=dtype).reshape(-1)

def frame_len(obj: Any) -> int:
    try:
        return int(len(obj))
    except Exception:
        return 0

def frame_index(obj: Any) -> Any | None:
    return getattr(obj, "index", None)

def frame_columns(obj: Any) -> list[str]:
    cols = getattr(obj, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []

def month_day_indices_from_index(index_like: Any) -> tuple[np.ndarray, np.ndarray]:
    idx_ns = index_to_ns_int64(index_like)
    if idx_ns is None or idx_ns.size <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    dt = idx_ns.astype("datetime64[ns]")
    month_idx = dt.astype("datetime64[M]").astype(np.int64, copy=False)
    day_idx = dt.astype("datetime64[D]").astype(np.int64, copy=False)
    return month_idx, day_idx

def frame_resolve_column(obj: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in frame_columns(obj):
        if str(col).strip().lower() == target:
            return col
    return None

def frame_has_column(obj: Any, name: str) -> bool:
    return frame_resolve_column(obj, name) is not None

def frame_empty(obj: Any) -> bool:
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

def frame_copy(obj: Any) -> Any:
    if obj is None:
        return None
    with contextlib.suppress(Exception):
        return obj.copy(deep=False)
    with contextlib.suppress(Exception):
        return obj.copy()
    return obj

def frame_column_numpy(obj: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    col = frame_resolve_column(obj, name)
    if col is None:
        raise KeyError(name)
    return to_numpy_1d(obj[col], dtype=dtype)

def frame_column_numpy_optional(obj: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray | None:
    with contextlib.suppress(Exception):
        return frame_column_numpy(obj, name, dtype=dtype)
    return None

def get_bar_time(obj: Any) -> Any:
    """Extract the latest timestamp from a frame's index or 'timestamp' column."""
    if frame_empty(obj):
        return None
    try:
        if frame_has_column(obj, "timestamp"):
            arr = frame_column_numpy(obj, "timestamp", dtype=object)
            if arr.size > 0:
                return arr[-1]
        idx = frame_index(obj)
        if idx is not None and len(idx) > 0:
            return idx[-1]
    except Exception:
        pass
    return None

def column_array(frame: Any, name: str) -> np.ndarray | None:
    """Safe extraction of a column as a numpy array."""
    try:
        return frame_column_numpy(frame, name)
    except Exception:
        return None

def frame_set_column(obj: Any, name: str, values: Any, *, dtype: Any = np.float32) -> bool:
    vals = np.asarray(values, dtype=dtype).reshape(-1)
    with contextlib.suppress(Exception):
        obj[str(name)] = vals
        return True
    data = getattr(obj, "_data", None)
    if isinstance(data, dict):
        key = str(name)
        n = frame_len(obj)
        data[key] = fit_len_array(vals, n, fill=0.0, dtype=dtype)
        cols = getattr(obj, "columns", None)
        if isinstance(cols, list) and key not in cols:
            cols.append(key)
        return True
    return False

def make_series(
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

def make_dataframe(
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
    return NumpyFrame(out_data, index=idx_obj)

    return NumpyFrame(out_data, index=idx_obj)

def frame_to_2d_float32(obj: Any, *, feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if is_dataframe(obj):
        cols = [str(c) for c in obj.columns]
        return obj.to_numpy(dtype=np.float32), cols
    if is_frame_like(obj):
        cols = frame_columns(obj)
        if not cols and feature_names:
            cols = feature_names
        mats = []
        for c in cols:
            try:
                mats.append(frame_column_numpy(obj, c, dtype=np.float32))
            except Exception:
                mats.append(np.zeros(frame_len(obj), dtype=np.float32))
        if not mats:
            return np.zeros((frame_len(obj), 0), dtype=np.float32), []
        return np.column_stack(mats), cols
    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    cols = feature_names if (feature_names and len(feature_names) == arr.shape[1]) else [f"f{i}" for i in range(arr.shape[1])]
    return arr, cols

def range_index(n: int) -> Any:
    return np.arange(max(0, int(n)), dtype=np.int64)

def to_datetime_index(values: Any) -> Any:
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

def concat_dataframes(items: list[Any]) -> Any | None:
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
        if not is_frame_like(out):
            return None
        cols: list[str] = []
        seen: set[str] = set()
        for frame in items:
            for col in frame_columns(frame):
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
            n = frame_len(frame)
            idx_obj = frame_index(frame)
            idx_arr = np.asarray(idx_obj).reshape(-1) if idx_obj is not None else np.arange(n, dtype=np.int64)
            idx_parts.append(_fit_len_any(idx_arr, n))
        for col in cols:
            chunks: list[np.ndarray] = []
            for frame in items:
                n = frame_len(frame)
                src_col = frame_resolve_column(frame, col)
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
        return NumpyFrame(out_data, index=idx_all)
    return out
