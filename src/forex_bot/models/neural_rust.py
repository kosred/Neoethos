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


def _to_float32_contig(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float32)
    elif hasattr(x, "to_numpy"):
        arr = x.to_numpy(dtype=np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _to_int32_contig(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.int32).reshape(-1)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr


class RustMLPExpert(ExpertModel):
    _model_cls = getattr(_fb, "MLPModel", None)

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_time_sec: int = 36000,
        device: str = "cpu",
        batch_size: int = 4096,
        idx: int = 1,
        **_: Any,
    ) -> None:
        if _fb is None:
            raise ImportError(
                "forex_bindings is not available; build the Rust bindings for neural models."
            ) from _BINDINGS_ERROR
        if self._model_cls is None:
            raise RuntimeError("Rust neural model unavailable for RustMLPExpert.")
        try:
            self._model = self._model_cls(
                idx=int(idx),
                hidden_dim=int(hidden_dim),
                n_layers=int(n_layers),
                dropout=float(dropout),
                lr=float(lr),
                max_time_sec=int(max_time_sec),
                device=str(device),
                batch_size=int(batch_size),
            )
        except Exception as exc:
            raise RuntimeError(f"Rust MLP init failed: {exc}") from exc

    def fit(self, x: Any, y: Any) -> None:
        try:
            self._model.fit(_to_float32_contig(x), _to_int32_contig(y))
        except Exception as exc:
            raise RuntimeError(f"Rust MLP training failed: {exc}") from exc

    def predict_proba(self, x: Any) -> np.ndarray:
        try:
            return np.asarray(self._model.predict_proba(_to_float32_contig(x)), dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Rust MLP prediction failed: {exc}") from exc

    def save(self, path: str) -> None:
        try:
            self._model.save(str(path))
        except Exception as exc:
            raise RuntimeError(f"Rust MLP save failed: {exc}") from exc

    def load(self, path: str) -> None:
        try:
            self._model.load(str(path))
        except Exception as exc:
            raise RuntimeError(f"Rust MLP load failed: {exc}") from exc


MLPExpert = RustMLPExpert
