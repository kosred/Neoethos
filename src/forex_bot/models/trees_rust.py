from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel

logger = logging.getLogger(__name__)

_BINDINGS_ERROR: Exception | None = None
try:
    import forex_bindings as _fb
except Exception as exc:  # pragma: no cover - import error depends on build env
    _fb = None
    _BINDINGS_ERROR = exc


def _python_fallback_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_TREE_RUST_FALLBACK", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _remap_labels_to_contiguous(y: pd.Series | np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int64)
    remapped = np.zeros_like(y_arr)
    remapped[y_arr == -1] = 0
    remapped[y_arr == 0] = 1
    remapped[y_arr == 1] = 2
    return remapped


def _to_float64_contig(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=np.float64, copy=False)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


class _RustTreeBase(ExpertModel):
    _model_cls = None
    _python_fallback_class_name: str | None = None

    def __init__(self, params: dict[str, Any] | None = None, idx: int = 1) -> None:
        self._params = dict(params or {})
        self._idx = int(idx)
        self._model = None
        self._fallback_model = None

        if _fb is not None and self._model_cls is not None:
            try:
                self._model = self._model_cls(idx=self._idx, params=self._params or None)
            except Exception as exc:
                logger.warning("Rust tree model init failed; will try Python fallback: %s", exc)

        if self._model is None and not _python_fallback_enabled():
            if _fb is None:
                raise ImportError(
                    "forex_bindings is not available; build the Rust bindings or enable "
                    "FOREX_BOT_TREE_RUST_FALLBACK=1 for Python fallback."
                ) from _BINDINGS_ERROR
            raise RuntimeError(
                f"Rust tree model unavailable for {self.__class__.__name__} and Python fallback is disabled."
            )

    def _ensure_python_fallback_model(self):
        if self._fallback_model is not None:
            return self._fallback_model
        cls_name = str(self._python_fallback_class_name or "").strip()
        if not cls_name:
            raise RuntimeError(f"No Python fallback class configured for {self.__class__.__name__}.")
        from . import trees as trees_py

        cls = getattr(trees_py, cls_name, None)
        if cls is None:
            raise ImportError(f"Python fallback class '{cls_name}' is unavailable.")
        kwargs = {"params": dict(self._params), "idx": int(self._idx)}
        try:
            self._fallback_model = cls(**kwargs)
        except Exception:
            kwargs.pop("idx", None)
            self._fallback_model = cls(**kwargs)
        return self._fallback_model

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        x_arr = _to_float64_contig(x)
        y_arr = _remap_labels_to_contiguous(y)

        if self._model is not None:
            try:
                self._model.fit(x_arr, y_arr)
                return
            except Exception as exc:
                logger.warning("Rust tree model training failed; switching to Python fallback: %s", exc)
                self._model = None

        if not _python_fallback_enabled():
            raise RuntimeError("Rust tree model training failed and Python fallback is disabled.")

        fallback = self._ensure_python_fallback_model()
        fallback.fit(x, y)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is not None:
            try:
                x_arr = _to_float64_contig(x)
                return np.asarray(self._model.predict_proba(x_arr), dtype=np.float32)
            except Exception as exc:
                logger.warning("Rust tree model prediction failed; switching to Python fallback: %s", exc)
                self._model = None

        if _python_fallback_enabled():
            try:
                fallback = self._ensure_python_fallback_model()
                return np.asarray(fallback.predict_proba(x), dtype=np.float32)
            except Exception as exc:
                logger.error("Python fallback tree prediction failed: %s", exc)

        return np.zeros((len(x), 3), dtype=np.float32)

    def save(self, path: str) -> None:
        if self._model is not None:
            try:
                self._model.save(path)
                return
            except Exception as exc:
                logger.warning("Rust tree model save failed; trying Python fallback save: %s", exc)
        if _python_fallback_enabled():
            fallback = self._ensure_python_fallback_model()
            fallback.save(path)
            return
        raise RuntimeError("Rust tree save failed and Python fallback is disabled.")

    def load(self, path: str) -> None:
        if self._model is not None:
            try:
                self._model.load(path)
                return
            except Exception as exc:
                logger.warning("Rust tree model load failed; trying Python fallback load: %s", exc)
                self._model = None
        if _python_fallback_enabled():
            fallback = self._ensure_python_fallback_model()
            fallback.load(path)
            return
        raise RuntimeError("Rust tree load failed and Python fallback is disabled.")


class RustLightGBMExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "LightGBMModel", None)
    _python_fallback_class_name = "LightGBMExpert"


class RustXGBoostExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostModel", None)
    _python_fallback_class_name = "XGBoostExpert"


class RustXGBoostRFExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostRFModel", None)
    _python_fallback_class_name = "XGBoostRFExpert"


class RustXGBoostDARTExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostDARTModel", None)
    _python_fallback_class_name = "XGBoostDARTExpert"


class RustCatBoostExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "CatBoostModel", None)
    _python_fallback_class_name = "CatBoostExpert"


class RustCatBoostAltExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "CatBoostAltModel", None)
    _python_fallback_class_name = "CatBoostAltExpert"


# Aliases for drop-in compatibility
LightGBMExpert = RustLightGBMExpert
XGBoostExpert = RustXGBoostExpert
XGBoostRFExpert = RustXGBoostRFExpert
XGBoostDARTExpert = RustXGBoostDARTExpert
CatBoostExpert = RustCatBoostExpert
CatBoostAltExpert = RustCatBoostAltExpert
