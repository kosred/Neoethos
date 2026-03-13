from __future__ import annotations

from typing import Any

import numpy as np

from .base import ExpertModel

_BINDINGS_ERROR: Exception | None = None
try:
    import forex_bindings as _fb
except Exception as exc:  # pragma: no cover - import error depends on build env
    _fb = None
    _BINDINGS_ERROR = exc


def _frame_columns(value: Any) -> list[str]:
    cols = getattr(value, "columns", None)
    if cols is None:
        return []
    try:
        return [str(col) for col in list(cols)]
    except Exception:
        return []


def _column_vector(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        try:
            arr = value.to_numpy(copy=False)
        except TypeError:
            arr = value.to_numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _to_named_matrix(value: Any, feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cols = _frame_columns(value)
    if cols:
        mats: list[np.ndarray] = []
        n_rows = 0
        out_cols: list[str] = []
        for col in cols:
            try:
                vec = _column_vector(value[col])  # type: ignore[index]
            except Exception:
                continue
            mats.append(vec)
            out_cols.append(str(col))
            n_rows = max(n_rows, int(vec.size))
        arr = np.zeros((n_rows, len(mats)), dtype=np.float64)
        for idx, vec in enumerate(mats):
            take = min(n_rows, int(vec.size))
            if take > 0:
                arr[:take, idx] = vec[:take]
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        return arr, out_cols

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
    elif hasattr(value, "to_numpy"):
        try:
            arr = value.to_numpy(dtype=np.float64, copy=False)
        except TypeError:
            arr = value.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    names = list(feature_names or [])
    if len(names) != int(arr.shape[1]):
        names = [f"feature_{idx}" for idx in range(int(arr.shape[1]))]
    return arr, names


def _to_int32_contig(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.int32).reshape(-1)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr


def _extract_symbol(metadata: Any) -> str | None:
    attrs = getattr(metadata, "attrs", None)
    if not isinstance(attrs, dict):
        return None
    symbol = str(attrs.get("symbol", "") or "").strip()
    return symbol or None


class RustGeneticExpert(ExpertModel):
    _model_cls = getattr(_fb, "GeneticModel", None)

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 10,
        max_indicators: int = 0,
        idx: int = 1,
        **_: Any,
    ) -> None:
        if _fb is None:
            raise ImportError(
                "forex_bindings is not available; build the Rust bindings for genetic models."
            ) from _BINDINGS_ERROR
        if self._model_cls is None:
            raise RuntimeError("Rust genetic model unavailable for RustGeneticExpert.")
        try:
            self._model = self._model_cls(
                idx=int(idx),
                population_size=int(population_size),
                generations=int(generations),
                max_indicators=int(max_indicators),
            )
        except Exception as exc:
            raise RuntimeError(f"Rust genetic init failed: {exc}") from exc

    def fit(self, x: Any, y: Any, metadata: Any | None = None) -> None:
        try:
            x_arr, x_cols = _to_named_matrix(x)
            kwargs: dict[str, Any] = {"feature_names": x_cols}
            if metadata is not None:
                meta_arr, meta_cols = _to_named_matrix(metadata)
                kwargs["metadata"] = meta_arr
                kwargs["metadata_columns"] = meta_cols
                symbol = _extract_symbol(metadata)
                if symbol:
                    kwargs["metadata_symbol"] = symbol
            self._model.fit(x_arr, _to_int32_contig(y), **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Rust genetic training failed: {exc}") from exc

    def predict_proba(self, x: Any, metadata: Any | None = None) -> np.ndarray:
        try:
            x_arr, x_cols = _to_named_matrix(x)
            kwargs: dict[str, Any] = {"feature_names": x_cols}
            if metadata is not None:
                meta_arr, meta_cols = _to_named_matrix(metadata)
                kwargs["metadata"] = meta_arr
                kwargs["metadata_columns"] = meta_cols
                symbol = _extract_symbol(metadata)
                if symbol:
                    kwargs["metadata_symbol"] = symbol
            return np.asarray(self._model.predict_proba(x_arr, **kwargs), dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Rust genetic prediction failed: {exc}") from exc

    def save(self, path: str) -> None:
        try:
            self._model.save(str(path))
        except Exception as exc:
            raise RuntimeError(f"Rust genetic save failed: {exc}") from exc

    def load(self, path: str) -> None:
        try:
            self._model.load(str(path))
        except Exception as exc:
            raise RuntimeError(f"Rust genetic load failed: {exc}") from exc


GeneticStrategyExpert = RustGeneticExpert
