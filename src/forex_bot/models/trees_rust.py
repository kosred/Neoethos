from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from .base import ExpertModel
from .label_utils import remap_labels_sell_neutral_buy

_BINDINGS_ERROR: Exception | None = None
try:
    import forex_bindings as _fb
except Exception as exc:  # pragma: no cover - import error depends on build env
    _fb = None
    _BINDINGS_ERROR = exc

def _remap_labels_to_contiguous(y: Any) -> np.ndarray:
    arr = remap_labels_sell_neutral_buy(y).astype(np.int32, copy=False)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr


def _to_float64_contig(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64)
    elif hasattr(x, "to_numpy"):
        arr = x.to_numpy(dtype=np.float64, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _len_rows(x: Any) -> int:
    try:
        return int(len(x))
    except Exception:
        arr = np.asarray(x)
        if arr.ndim >= 1:
            return int(arr.shape[0])
        return 0


class _RustTreeBase(ExpertModel):
    _model_cls = None
    _artifact_name = "model.bin"

    def __init__(self, params: dict[str, Any] | None = None, idx: int = 1) -> None:
        self._params = dict(params or {})
        self._idx = int(idx)
        self._model = None

        if _fb is not None and self._model_cls is not None:
            try:
                self._model = self._model_cls(idx=self._idx, params=self._params or None)
            except Exception as exc:
                raise RuntimeError(
                    f"Rust tree model init failed for {self.__class__.__name__}: {exc}"
                ) from exc

        if self._model is None:
            if _fb is None:
                raise ImportError(
                    "forex_bindings is not available; build the Rust bindings for tree models."
                ) from _BINDINGS_ERROR
            raise RuntimeError(f"Rust tree model unavailable for {self.__class__.__name__}.")

    def fit(self, x: Any, y: Any) -> None:
        x_arr = _to_float64_contig(x)
        y_arr = _remap_labels_to_contiguous(y)

        if self._model is not None:
            try:
                self._model.fit(x_arr, y_arr)
                return
            except Exception as exc:
                raise RuntimeError(f"Rust tree model training failed: {exc}") from exc
        raise RuntimeError("Rust tree model is unavailable for training.")

    def predict_proba(self, x: Any) -> np.ndarray:
        if self._model is not None:
            try:
                x_arr = _to_float64_contig(x)
                return np.asarray(self._model.predict_proba(x_arr), dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(f"Rust tree model prediction failed: {exc}") from exc
        raise RuntimeError(
            f"Rust tree model is unavailable for prediction (rows={_len_rows(x)})."
        )

    def save(self, path: str) -> None:
        if self._model is not None:
            try:
                self._model.save(str(self._artifact_path(path, create_parent=True)))
                return
            except Exception as exc:
                raise RuntimeError(f"Rust tree model save failed: {exc}") from exc
        raise RuntimeError("Rust tree model is unavailable for save.")

    def load(self, path: str) -> None:
        if self._model is not None:
            try:
                self._model.load(str(self._artifact_path(path, create_parent=False)))
                return
            except Exception as exc:
                raise RuntimeError(f"Rust tree model load failed: {exc}") from exc
        raise RuntimeError("Rust tree model is unavailable for load.")

    def _artifact_path(self, path: str, *, create_parent: bool) -> Path:
        base = Path(path)
        if base.exists() and base.is_dir():
            target = base / self._artifact_name
        elif base.suffix:
            target = base
        else:
            target = base / self._artifact_name
        if create_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target


class RustLightGBMExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "LightGBMModel", None)
    _artifact_name = "lightgbm.txt"


class RustXGBoostExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostModel", None)
    _artifact_name = "xgboost.json"


class RustXGBoostRFExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostRFModel", None)
    _artifact_name = "xgboost_rf.json"


class RustXGBoostDARTExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostDARTModel", None)
    _artifact_name = "xgboost_dart.json"


class RustCatBoostExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "CatBoostModel", None)
    _artifact_name = "catboost.cbm"


class RustCatBoostAltExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "CatBoostAltModel", None)
    _artifact_name = "catboost_alt.cbm"


# Aliases for drop-in compatibility
LightGBMExpert = RustLightGBMExpert
XGBoostExpert = RustXGBoostExpert
XGBoostRFExpert = RustXGBoostRFExpert
XGBoostDARTExpert = RustXGBoostDARTExpert
CatBoostExpert = RustCatBoostExpert
CatBoostAltExpert = RustCatBoostAltExpert

