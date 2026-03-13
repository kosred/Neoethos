from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.distributed as dist
import yaml

from ..core.config import Settings
from ..core.system import normalize_device_preference
from ..domain.events import PreparedDataset
from ..models.base import ExpertModel, detect_feature_drift
from ..strategy.fast_backtest import (
    fast_evaluate_strategy,
    infer_pip_metrics,
    infer_sl_tp_pips_auto,
)
from .ensemble import MetaBlender
from .evaluation import probs_to_signals, prop_backtest
from .optimization import HyperparameterOptimizer

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Services
from .benchmark_service import BenchmarkService
from .calibration import ProbabilityCalibrator
from .conformal import ConformalClassifierGate
from .evaluation_service import EvaluationService
from .model_factory import ModelFactory
from .persistence_service import PersistenceService

logger = logging.getLogger(__name__)
try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore

_PANDAS_FREE_MODEL_ALLOWLIST = {
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
    # Numpy-native linear experts (no tabular-module dependency at fit/predict).
    "elasticnet",
    "bayes_logit",
    "online_pa",
    "online_hoeffding",
    "vw",
}

_PANDAS_FREE_RUST_TREE_REQUIRED = {
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
}

_RUST_TREE_BINDING_CLASSES = {
    "lightgbm": "LightGBMModel",
    "xgboost": "XGBoostModel",
    "xgboost_rf": "XGBoostRFModel",
    "xgboost_dart": "XGBoostDARTModel",
    "catboost": "CatBoostModel",
    "catboost_alt": "CatBoostAltModel",
}


def _rust_tree_binding_available(model_name: str) -> bool:
    cls_name = _RUST_TREE_BINDING_CLASSES.get(str(model_name))
    if not cls_name or _fb is None:
        return False
    return bool(hasattr(_fb, cls_name))


def _is_dataframe(value: Any) -> bool:
    return bool(
        hasattr(value, "columns")
        and hasattr(value, "index")
        and callable(getattr(value, "to_numpy", None))
    )


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


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))


class _NumpyFrame:
    """Minimal frame container used for non-dataframe-module slices/alignment."""

    def __init__(self, data: dict[str, Any], index: Any, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def shape(self) -> tuple[int, int]:
        return int(len(self.index)), int(len(self.columns))

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def to_numpy(self, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if not self.columns:
            out = np.zeros((len(self.index), 0), dtype=np.float32)
        else:
            out = np.column_stack([np.asarray(self._data[c]).reshape(-1) for c in self.columns])
        if dtype is not None:
            out = np.asarray(out, dtype=dtype)
        elif copy:
            out = np.array(out, copy=True)
        return out


def _frame_columns(value: Any) -> list[str]:
    cols = getattr(value, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_resolve_column(value: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(value):
        if str(col).strip().lower() == target:
            return col
    return None


def _frame_has_column(value: Any, name: str) -> bool:
    return _frame_resolve_column(value, name) is not None


def _frame_index(value: Any) -> Any | None:
    return getattr(value, "index", None)


def _frame_extract_column(value: Any, name: str) -> np.ndarray | None:
    target = str(name).strip().lower()
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).strip().lower() == target:
                return np.asarray(v).reshape(-1)
        return None
    col = _frame_resolve_column(value, name)
    if col is None or not hasattr(value, "__getitem__"):
        return None
    try:
        raw = value[col]  # type: ignore[index]
        arr = raw.to_numpy(copy=False) if hasattr(raw, "to_numpy") else np.asarray(raw)
        return np.asarray(arr).reshape(-1)
    except Exception:
        return None


def _index_to_int64(index_like: Any) -> np.ndarray | None:
    if index_like is None:
        return None
    try:
        if hasattr(index_like, "asi8"):
            arr = np.asarray(index_like.asi8, dtype=np.int64).reshape(-1)
            return arr if arr.size > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        pass
    try:
        arr = np.asarray(index_like).reshape(-1)
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


def _rust_time_index_arrays(
    index_like: Any,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if n_rows <= 0 or _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
        return None
    idx_ns = _index_to_int64(index_like)
    if idx_ns is None or idx_ns.size != int(n_rows):
        return None
    try:
        _unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(np.asarray(idx_ns, dtype=np.int64))
    except Exception:
        return None
    month_arr = np.asarray(month_idx, dtype=np.int64).reshape(-1)
    day_arr = np.asarray(day_idx, dtype=np.int64).reshape(-1)
    if month_arr.size != int(n_rows) or day_arr.size != int(n_rows):
        return None
    return month_arr, day_arr


def _rust_sorted_index_order(index_like: Any) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "sorted_index_order"):
        return None
    idx_ns = _index_to_int64(index_like)
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
    idx_ns = _index_to_int64(index_like)
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


def _column_index_mapping(
    src_names: list[str],
    dst_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    dst_lookup = {str(name).strip().lower(): i for i, name in enumerate(dst_names)}
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for src_i, raw in enumerate(src_names):
        dst_i = dst_lookup.get(str(raw).strip().lower())
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
    elif src.ndim > 2:
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


def _index_is_monotonic_increasing(index_like: Any) -> bool:
    if index_like is None:
        return True
    with contextlib.suppress(Exception):
        return bool(getattr(index_like, "is_monotonic_increasing"))
    idx = _index_to_int64(index_like)
    if idx is None or idx.size <= 1:
        return True
    return bool(np.all(idx[1:] >= idx[:-1]))


def _slice_frame_rows(value: Any, rows: np.ndarray) -> _NumpyFrame:
    cols = _frame_columns(value)
    row_sel = np.asarray(rows).reshape(-1)
    if row_sel.dtype == bool:
        row_idx = np.flatnonzero(row_sel)
    else:
        row_idx = row_sel.astype(np.int64, copy=False)

    n_rows = int(len(value)) if hasattr(value, "__len__") else 0
    data: dict[str, np.ndarray] = {}
    for col in cols:
        vec = _frame_extract_column(value, col)
        if vec is None:
            continue
        if row_sel.dtype == bool:
            if vec.size != row_sel.size:
                mask = np.zeros(vec.size, dtype=bool)
                take = min(vec.size, row_sel.size)
                if take > 0:
                    mask[:take] = row_sel[:take]
                data[str(col)] = vec[mask]
            else:
                data[str(col)] = vec[row_sel]
        else:
            safe_idx = row_idx
            if vec.size > 0:
                safe_idx = np.clip(row_idx, 0, vec.size - 1)
            data[str(col)] = vec[safe_idx] if safe_idx.size > 0 else np.asarray(vec[:0])

    idx_obj = _frame_index(value)
    if idx_obj is None:
        if row_sel.dtype == bool:
            out_index = np.arange(int(row_idx.size), dtype=np.int64)
        else:
            out_index = row_idx.astype(np.int64, copy=False)
    else:
        idx_arr = np.asarray(idx_obj).reshape(-1)
        if row_sel.dtype == bool:
            if idx_arr.size != row_sel.size:
                mask = np.zeros(idx_arr.size, dtype=bool)
                take = min(idx_arr.size, row_sel.size)
                if take > 0:
                    mask[:take] = row_sel[:take]
                out_index = idx_arr[mask]
            else:
                out_index = idx_arr[row_sel]
        else:
            if idx_arr.size > 0:
                safe_idx = np.clip(row_idx, 0, idx_arr.size - 1)
                out_index = idx_arr[safe_idx]
            else:
                out_index = np.asarray([], dtype=np.int64)
    attrs = getattr(value, "attrs", None)
    if len(out_index) != int(row_idx.size) and row_sel.dtype != bool:
        out_index = row_idx.astype(np.int64, copy=False)
    if not cols and n_rows <= 0:
        out_index = np.asarray([], dtype=np.int64)
    return _NumpyFrame(data, out_index, attrs=(dict(attrs) if isinstance(attrs, dict) else None))


def _frame_to_2d_float32(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        try:
            arr = values.to_numpy(dtype=np.float32, copy=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
        except Exception:
            pass
    cols = _frame_columns(values)
    if cols and hasattr(values, "__getitem__"):
        mats: list[np.ndarray] = []
        n_rows = 0
        for col in cols:
            vec = _frame_extract_column(values, col)
            if vec is None:
                continue
            with contextlib.suppress(Exception):
                kind = np.asarray(vec).dtype.kind
                if kind not in {"i", "u", "f", "b"}:
                    continue
            try:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            mats.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            n_rows = max(n_rows, int(arr.size))
        if mats:
            out = np.zeros((n_rows, len(mats)), dtype=np.float32)
            for j, vec in enumerate(mats):
                take = min(n_rows, int(vec.size))
                if take > 0:
                    out[:take, j] = vec[:take]
            return out
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _feature_names(values: Any, n_cols: int) -> list[str]:
    cols = _frame_columns(values)
    if cols and len(cols) == int(n_cols):
        return cols
    return [f"f{i}" for i in range(max(0, int(n_cols)))]


def _find_entrypoint_script() -> Path:
    """
    Locate the repo's `forex-ai.py` entrypoint.

    This trainer sometimes runs from installed packages or different working directories, so we
    search upward from the current file rather than assuming a fixed directory depth.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "forex-ai.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate entrypoint script `forex-ai.py` from {here}")


# Optional FSDP imports
try:  # pragma: no cover
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
except Exception:  # pragma: no cover
    FSDP = None
    ShardingStrategy = None
    MixedPrecision = None


class ModelTrainer:
    """
    Orchestrates model training using micro-services.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.optimizer = HyperparameterOptimizer(settings)

        self.models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        self.logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))

        # Services
        self.evaluator = EvaluationService(settings, self.models_dir)
        self.factory = ModelFactory(settings, self.models_dir)
        self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        self.benchmarker = BenchmarkService()

        # State
        self.models: dict[str, ExpertModel] = {}
        self.meta_blender: MetaBlender | None = None
        self.run_summary: dict[str, Any] = {}
        self.incremental_stats: dict[str, Any] = {}

        # Load history
        self.run_summary = self.persistence.load_run_summary()
        self._historical_durations = self.run_summary.get("train_durations_sec", {})
        self._historical_samples = self.run_summary.get("train_samples", None)
        self._historical_hardware = self.run_summary.get("train_hardware", {})
        self.incremental_stats = self.persistence.load_incremental_stats()
        # Distributed state
        self.distributed_enabled = False
        self.rank = 0
        self.world_size = 1

        self._maybe_init_distributed()

    def _maybe_init_distributed(self) -> None:
        """
        Initialize torch.distributed if enabled via settings.models.enable_ddp/fsdp and world_size > 1.
        Safe no-op if torch or distributed backend is unavailable.
        """
        try:
            use_dist = bool(
                getattr(self.settings.models, "enable_ddp", False) or getattr(self.settings.models, "enable_fsdp", False)
            )
            self.world_size = int(getattr(self.settings.models, "ddp_world_size", 1) or 1)
            if not use_dist or self.world_size <= 1:
                return
            if not dist.is_available():
                logger.warning("torch.distributed not available; running single-process.")
                return
            if dist.is_initialized():
                self.distributed_enabled = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                return
            # Default to env:// init (expect MASTER_ADDR/PORT set by launcher)
            backend = "gloo" if os.name == "nt" else ("nccl" if torch.cuda.is_available() else "gloo")
            dist.init_process_group(backend=backend)
            self.distributed_enabled = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            logger.info(f"Distributed initialized: rank {self.rank}/{self.world_size - 1}")
        except Exception as exc:
            logger.warning(f"Failed to initialize distributed; falling back to single process: {exc}")

    @staticmethod
    def _strict_model_check_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_STRICT_MODEL_CHECK", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _roundtrip_check_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_MODEL_ROUNDTRIP", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _pandas_free_enabled() -> bool:
        raw_env = os.environ.get("FOREX_BOT_PANDAS_FREE")
        if raw_env is not None and str(raw_env).strip() != "":
            raw = str(raw_env).strip().lower()
            return raw in {"1", "true", "yes", "on"}
        # Keep frame-native default behavior unless explicitly disabled.
        return True

    @staticmethod
    def _rust_only_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
        if profile.startswith("rust"):
            return True
        tree_backend = str(os.environ.get("FOREX_BOT_TREE_BACKEND", "") or "").strip().lower()
        return tree_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}

    @staticmethod
    def _pandas_free_strict_enabled() -> bool:
        raw = str(os.environ.get("FOREX_BOT_PANDAS_FREE_STRICT", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _memmap_dataset_complete(dataset_dir: Path | str | None) -> bool:
        if dataset_dir is None:
            return False
        try:
            root = Path(dataset_dir)
        except Exception:
            return False
        required = ("X.npy", "y.npy", "columns.json")
        try:
            return bool(root.exists() and all((root / name).exists() for name in required))
        except Exception:
            return False

    @staticmethod
    def _fit_accepts_metadata(model: ExpertModel) -> bool:
        try:
            sig = inspect.signature(model.fit)
            if "metadata" in sig.parameters:
                return True
            return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            return False

    @staticmethod
    def _predict_kwargs(model: ExpertModel, metadata: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if metadata is None:
            return kwargs
        try:
            sig = inspect.signature(model.predict_proba)
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "metadata" in sig.parameters or has_kwargs:
                kwargs["metadata"] = metadata
        except Exception:
            pass
        return kwargs

    @staticmethod
    def _coerce_feature_container_float32(values: Any) -> Any:
        if values is None:
            return values
        if _is_dataframe(values):
            out = values
            try:
                float_cols = [c for c in out.columns if getattr(out[c], "dtype", np.dtype("O")).kind == "f"]
                if float_cols:
                    out[float_cols] = out[float_cols].astype(np.float32)
            except Exception:
                pass
            return out
        if _is_frame_like(values):
            return ModelTrainer._align_feature_frame(values, _frame_columns(values))
        return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _metadata_to_numpy_frame(metadata: Any) -> Any:
        if metadata is None:
            return None
        if isinstance(metadata, _NumpyFrame):
            return metadata
        if not _is_frame_like(metadata):
            return metadata

        cols = _frame_columns(metadata)
        n_rows = int(len(metadata)) if hasattr(metadata, "__len__") else 0
        if not cols or n_rows <= 0:
            return metadata

        data: dict[str, np.ndarray] = {}
        for col in cols:
            vec = _frame_extract_column(metadata, col)
            if vec is None:
                continue
            out = np.asarray(vec).reshape(-1)
            if out.size < n_rows:
                pad = np.zeros(n_rows, dtype=out.dtype if out.size > 0 else np.float32)
                if out.size > 0:
                    pad[: out.size] = out
                out = pad
            elif out.size > n_rows:
                out = out[:n_rows]
            data[str(col)] = out

        idx_obj = _frame_index(metadata)
        idx_arr = np.asarray(idx_obj).reshape(-1) if idx_obj is not None else np.arange(n_rows, dtype=np.int64)
        if idx_arr.size < n_rows:
            idx_arr = np.arange(n_rows, dtype=np.int64)
        else:
            idx_arr = idx_arr[:n_rows]
        attrs = getattr(metadata, "attrs", None)
        return _NumpyFrame(data, idx_arr, attrs=(dict(attrs) if isinstance(attrs, dict) else None))

    @staticmethod
    def _persist_metadata_artifact(metadata: Any, path: Path) -> Path | None:
        if metadata is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            joblib.dump(metadata, path)
            return path
        except Exception as exc:
            try:
                compact = ModelTrainer._metadata_to_numpy_frame(metadata)
                joblib.dump(compact, path)
                logger.debug("Persisted metadata via compact frame fallback: %s", path)
                return path
            except Exception as exc_fallback:
                logger.warning(
                    "Failed to persist metadata artifact: %s (fallback failed: %s)",
                    exc,
                    exc_fallback,
                )
                return None

    def _smoke_predict(
        self,
        *,
        model_name: str,
        model: ExpertModel,
        sample_x: Any,
        sample_meta: Any | None = None,
    ) -> tuple[bool, str]:
        if sample_x is None or len(sample_x) == 0:
            return False, "empty sample for smoke test"
        try:
            kwargs = self._predict_kwargs(model, sample_meta)
            probs = self._pad_probs(model.predict_proba(sample_x, **kwargs))
        except Exception as exc:
            return False, f"inference exception: {exc}"
        if probs.ndim != 2:
            return False, f"invalid probability rank: {probs.ndim}"
        if probs.shape[0] != len(sample_x):
            return False, f"row mismatch: got {probs.shape[0]} expected {len(sample_x)}"
        if probs.shape[1] < 2:
            return False, f"invalid probability width: {probs.shape[1]}"
        if not np.all(np.isfinite(probs)):
            return False, "non-finite probabilities detected"
        return True, f"{model_name} smoke check ok"

    def _roundtrip_smoke_check(
        self,
        *,
        model_name: str,
        idx: int,
        model: ExpertModel,
        sample_x: Any,
        sample_meta: Any | None = None,
    ) -> tuple[bool, str]:
        if not self._roundtrip_check_enabled():
            return True, "roundtrip disabled"
        stamp = f"{int(time.time() * 1_000_000)}_{os.getpid()}"
        tmp_dir = self.models_dir / "_healthcheck" / f"{model_name}_{idx}_{stamp}"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(tmp_dir))
            probe_factory = ModelFactory(self.settings, tmp_dir)
            reloaded = probe_factory.create_model(model_name, {}, idx)
            if not hasattr(reloaded, "load"):
                return False, "reloaded model has no load()"
            reloaded.load(str(tmp_dir))
            ok, reason = self._smoke_predict(
                model_name=model_name,
                model=reloaded,
                sample_x=sample_x,
                sample_meta=sample_meta,
            )
            if not ok:
                return False, f"roundtrip inference failed: {reason}"
            return True, "roundtrip smoke check ok"
        except Exception as exc:
            return False, f"roundtrip exception: {exc}"
        finally:
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self.distributed_enabled = False
            self.rank = 0
            self.world_size = 1

    def estimate_time_for_dataset(
        self,
        X: Any,
        n_samples: int,
        context: str = "incremental",
        simulate_gpu: str | None = None,
    ) -> float:
        """
        Estimate runtime for the current enabled models on a dataset of size n_samples.
        Uses probes, historical stats, and incremental timings. X is only used for probes.
        """
        device = (
            "cuda"
            if bool(getattr(self.settings.system, "enable_gpu", False))
            else str(getattr(self.settings.system, "device", "cpu"))
        )
        probe_n = min(10_000, int(len(X)))
        probe_x = self._slice_rows(X, np.arange(probe_n, dtype=np.int64))
        bench_result = self.benchmarker.run_micro_benchmark(
            X, np.zeros(min(len(X), 10), dtype=np.int8), device if simulate_gpu is None else "cuda"
        )
        est_time = self.benchmarker.estimate_time(
            self._get_enabled_models(),
            n_samples,
            bench_result,
            bool(getattr(self.settings.system, "enable_gpu", False)),
            int(getattr(self.settings.system, "num_gpus", 1)),
            context=context,
            historical_durations=self._historical_durations,
            historical_n=self._historical_samples,
            historical_gpu=(
                self._historical_hardware.get("enable_gpu"),
                self._historical_hardware.get("num_gpus", 1),
            ),
            incremental_stats=self.incremental_stats,
            probe_kwargs={
                "X": probe_x,
                "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                "device": device,
                "steps": 6 if context == "incremental" else 8,
            },
            simulate_gpu=simulate_gpu,
        )
        return est_time

    @staticmethod
    def _read_int_env(name: str, default: int | None = None) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except Exception:
            return default

    @staticmethod
    def _read_float_env(name: str, default: float | None = None) -> float | None:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(str(raw).strip())
        except Exception:
            return default

    @staticmethod
    def _slice_rows(values: Any, rows: np.ndarray) -> Any:
        if values is None:
            return None
        arr_rows = np.asarray(rows)
        if _is_dataframe(values):
            if arr_rows.dtype == bool:
                with contextlib.suppress(Exception):
                    return values.loc[arr_rows]
            idx = np.asarray(arr_rows, dtype=np.int64).reshape(-1)
            with contextlib.suppress(Exception):
                return values.take(idx)
            with contextlib.suppress(Exception):
                base_idx = np.asarray(getattr(values, "index")).reshape(-1)
                return values.loc[base_idx[idx]]
        if _is_frame_like(values):
            return _slice_frame_rows(values, arr_rows)
        arr = np.asarray(values)
        return arr[arr_rows]

    @staticmethod
    def _sample_rows(values: Any, n: int, *, random_state: int = 42) -> Any:
        total = int(len(values))
        n = max(0, min(int(n), total))
        if n <= 0:
            return ModelTrainer._slice_rows(values, np.arange(0, dtype=np.int64))
        if n >= total:
            return values
        if hasattr(values, "sample"):
            try:
                return values.sample(n=n, random_state=random_state)
            except Exception:
                pass
        rng = np.random.default_rng(int(random_state))
        rows = np.sort(rng.choice(total, size=n, replace=False)).astype(np.int64, copy=False)
        return ModelTrainer._slice_rows(values, rows)

    @staticmethod
    def _as_float32_matrix(values: Any) -> np.ndarray:
        arr = _frame_to_2d_float32(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        out = np.asarray(arr, dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    @staticmethod
    def _utc_iso_from_ns(ns: int) -> str:
        try:
            dt = np.datetime64(int(ns), "ns")
            return str(np.datetime_as_string(dt, unit="ns", timezone="UTC"))
        except Exception:
            return str(int(ns))

    @staticmethod
    def _month_day_indices_from_index(index_like: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        month_idx = np.zeros(n_rows, dtype=np.int64)
        day_idx = np.zeros(n_rows, dtype=np.int64)
        if n_rows <= 0:
            return month_idx, day_idx
        rust = _rust_time_index_arrays(index_like, n_rows)
        if rust is not None:
            return rust
        idx_ns = _index_to_int64(index_like)
        if idx_ns is not None and idx_ns.size == n_rows:
            try:
                if idx_ns.size > 0 and int(np.max(np.abs(idx_ns))) > 10**14:
                    dt = np.asarray(idx_ns, dtype=np.int64).astype("datetime64[ns]")
                    month_idx = dt.astype("datetime64[M]").astype(np.int64, copy=False)
                    day_idx = dt.astype("datetime64[D]").astype(np.int64, copy=False)
                    return month_idx, day_idx
            except Exception:
                pass
        try:
            arr = np.asarray(index_like).reshape(-1)
            if arr.size != n_rows:
                return month_idx, day_idx
            if np.issubdtype(arr.dtype, np.datetime64):
                dt = arr.astype("datetime64[ns]")
            elif arr.dtype.kind in {"i", "u"}:
                dt = arr.astype(np.int64, copy=False).astype("datetime64[ns]")
            elif arr.dtype.kind == "f":
                ints = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
                dt = ints.astype("datetime64[ns]")
            else:
                return month_idx, day_idx
            month_idx = dt.astype("datetime64[M]").astype(np.int64, copy=False)
            day_idx = dt.astype("datetime64[D]").astype(np.int64, copy=False)
        except Exception:
            return month_idx, day_idx
        return month_idx, day_idx

    def _selected_features_path(self) -> Path:
        return self.models_dir / "selected_features.json"

    def _selected_features_by_regime_path(self) -> Path:
        return self.models_dir / "selected_features_by_regime.json"

    def _load_selected_features(self) -> list[str]:
        path = self._selected_features_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(c) for c in payload if str(c).strip()]
        except Exception as exc:
            logger.debug("Failed to load selected features: %s", exc)
        return []

    def _save_selected_features(self, cols: list[str]) -> None:
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self._selected_features_path().write_text(json.dumps(list(cols), indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist selected features: %s", exc)

    def _save_selected_features_by_regime(self, mapping: dict[str, list[str]]) -> None:
        try:
            path = self._selected_features_by_regime_path()
            self.models_dir.mkdir(parents=True, exist_ok=True)
            if not mapping:
                if path.exists():
                    path.unlink()
                return
            payload = {
                str(k): [str(c) for c in list(v or []) if str(c).strip()]
                for k, v in mapping.items()
                if str(k).strip()
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist per-regime selected features: %s", exc)

    @staticmethod
    def _align_feature_frame(df: Any, cols: list[str]) -> Any:
        if df is None:
            return np.zeros((0, len(cols)), dtype=np.float32)
        if not cols:
            return df

        arr = ModelTrainer._as_float32_matrix(df)
        base_names = _frame_columns(df)
        if len(base_names) != int(arr.shape[1]):
            base_names = _feature_names(df, arr.shape[1])
        src_cols, dst_cols = _column_index_mapping(base_names, cols)
        aligned = _align_feature_matrix(
            arr,
            src_cols,
            dst_cols,
            dst_width=len(cols),
        )
        if _is_frame_like(df):
            n_rows = int(aligned.shape[0])
            data = {str(col): np.asarray(aligned[:, j], dtype=np.float32).reshape(-1) for j, col in enumerate(cols)}
            idx_obj = _frame_index(df)
            idx_arr = np.asarray(idx_obj).reshape(-1) if idx_obj is not None else np.arange(n_rows, dtype=np.int64)
            if idx_arr.size < n_rows:
                idx_arr = np.arange(n_rows, dtype=np.int64)
            else:
                idx_arr = idx_arr[:n_rows]
            attrs = getattr(df, "attrs", None)
            return _NumpyFrame(data, idx_arr, attrs=(dict(attrs) if isinstance(attrs, dict) else None))
        return aligned

    @staticmethod
    def _numeric_feature_matrix(X: Any) -> tuple[np.ndarray, list[str]]:
        cols = _frame_columns(X)
        if cols and hasattr(X, "__getitem__"):
            mats: list[np.ndarray] = []
            names: list[str] = []
            n_rows = int(len(X))
            for col in cols:
                vec = _frame_extract_column(X, col)
                if vec is None:
                    continue
                vec = np.asarray(vec).reshape(-1)
                if vec.size < n_rows:
                    padded = np.zeros(n_rows, dtype=np.float32)
                    if vec.size > 0:
                        with contextlib.suppress(Exception):
                            padded[: vec.size] = np.asarray(vec, dtype=np.float32)
                    vec = padded
                elif vec.size > n_rows:
                    vec = vec[:n_rows]
                kind = np.asarray(vec).dtype.kind
                if kind not in {"i", "u", "f", "b"}:
                    continue
                with contextlib.suppress(Exception):
                    vec = np.nan_to_num(np.asarray(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    mats.append(vec.reshape(-1))
                    names.append(str(col))
            if mats:
                return np.column_stack(mats).astype(np.float32, copy=False), names
        arr = ModelTrainer._as_float32_matrix(X)
        return arr, _feature_names(X, arr.shape[1])

    @staticmethod
    def _canon_labels(y: Any) -> np.ndarray:
        arr = np.asarray(y, dtype=int)
        arr = np.where(arr == -1, 2, arr).astype(int, copy=False)
        return np.clip(arr, 0, 2)

    def _l1_regime_mask(self, X: Any) -> np.ndarray:
        # 0=range, 1=neutral, 2=trend
        n = int(len(X))
        if n == 0:
            return np.zeros(0, dtype=np.int8)
        adx_col = _frame_resolve_column(X, "adx")
        if adx_col is None:
            for c in _frame_columns(X):
                if str(c).strip().lower().endswith("_adx"):
                    adx_col = str(c)
                    break
        if adx_col is None:
            return np.ones(n, dtype=np.int8)
        adx_vals = _frame_extract_column(X, adx_col)
        if adx_vals is None:
            return np.ones(n, dtype=np.int8)
        adx = np.asarray(adx_vals, dtype=np.float32).reshape(-1)
        if adx.size < n:
            padded = np.zeros(n, dtype=np.float32)
            if adx.size > 0:
                padded[: adx.size] = adx
            adx = padded
        elif adx.size > n:
            adx = adx[:n]
        adx = np.nan_to_num(adx, nan=0.0, posinf=0.0, neginf=0.0)
        trend_thr = float(getattr(self.settings.risk, "regime_adx_trend", 25.0) or 25.0)
        range_thr = float(getattr(self.settings.risk, "regime_adx_range", 20.0) or 20.0)
        mask = np.ones(n, dtype=np.int8)
        mask[adx >= trend_thr] = 2
        mask[adx <= range_thr] = 0
        return mask

    def _apply_l1_feature_selection(
        self,
        X_fit: Any,
        y_fit: Any,
        X_eval: Any,
    ) -> tuple[Any, Any, list[str]]:
        enabled = bool(getattr(self.settings.models, "l1_feature_selection_enabled", True))
        source_cols = _frame_columns(X_fit)
        if not source_cols:
            source_cols = _feature_names(X_fit, self._as_float32_matrix(X_fit).shape[1])
        if not enabled:
            cols = list(source_cols)
            self._save_selected_features(cols)
            self._save_selected_features_by_regime({})
            return X_fit, X_eval, cols
        if len(X_fit) < 200:
            cols = list(source_cols)
            self._save_selected_features(cols)
            self._save_selected_features_by_regime({})
            return X_fit, X_eval, cols

        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            cols = list(source_cols)
            self._save_selected_features(cols)
            self._save_selected_features_by_regime({})
            logger.warning("L1 feature selection skipped: scikit-learn LogisticRegression unavailable.")
            return X_fit, X_eval, cols

        x_num, x_num_cols = self._numeric_feature_matrix(X_fit)
        if x_num.shape[1] <= 0:
            cols = list(source_cols)
            self._save_selected_features(cols)
            self._save_selected_features_by_regime({})
            return X_fit, X_eval, cols

        y_arr = self._canon_labels(y_fit)
        per_regime = bool(getattr(self.settings.models, "l1_feature_selection_per_regime", True))
        reg_mask = self._l1_regime_mask(X_fit) if per_regime else np.ones(len(X_fit), dtype=np.int8)

        try:
            sample_lim = int(getattr(self.settings.models, "l1_feature_selection_sample_limit", 200_000) or 200_000)
        except Exception:
            sample_lim = 200_000
        sample_lim = max(500, sample_lim)

        c_val = float(getattr(self.settings.models, "l1_feature_selection_c", 0.2) or 0.2)
        c_val = max(1e-4, c_val)
        score = np.zeros(x_num.shape[1], dtype=np.float64)
        regime_scores: dict[int, np.ndarray] = {}

        for rid in sorted(set(reg_mask.tolist())):
            idx = np.where(reg_mask == int(rid))[0]
            if idx.size < 150:
                continue
            x_r = x_num[idx]
            y_r = y_arr[idx]
            if len(np.unique(y_r)) < 2:
                continue
            if len(x_r) > sample_lim:
                take = np.linspace(0, len(x_r) - 1, num=sample_lim, dtype=int)
                x_r = x_r[take]
                y_r = y_r[take]
            try:
                clf = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=c_val,
                    max_iter=400,
                    multi_class="ovr",
                    class_weight="balanced",
                    random_state=42,
                )
                clf.fit(np.asarray(x_r, dtype=np.float32), y_r)
                coef = np.abs(np.asarray(clf.coef_, dtype=np.float64))
                if coef.ndim == 2:
                    coef = coef.sum(axis=0)
                score += coef
                regime_scores[int(rid)] = np.asarray(coef, dtype=np.float64)
            except Exception as exc:
                logger.debug("L1 feature selection regime %s failed: %s", rid, exc)

        cols = list(x_num_cols)
        # Fallback ranking by variance if L1 couldn't fit.
        if not np.any(np.isfinite(score)) or float(np.nanmax(score)) <= 0.0:
            with contextlib.suppress(Exception):
                score = np.asarray(np.var(x_num, axis=0), dtype=np.float64)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        must_keep = [
            c
            for c in ("open", "high", "low", "close", "volume", "base_signal", "rsi", "macd_hist", "adx", "atr14")
            if c in source_cols
        ]

        min_k = int(getattr(self.settings.models, "l1_feature_selection_min_features", 20) or 20)
        max_k = int(getattr(self.settings.models, "l1_feature_selection_max_features", 256) or 256)
        min_k = max(5, min_k)
        max_k = max(min_k, max_k) if max_k > 0 else 0

        def _ordered_from_score(score_vec: np.ndarray) -> list[str]:
            order = _rust_rank_scores_desc(score_vec)
            if order is None:
                order = np.argsort(-np.asarray(score_vec, dtype=np.float64))
            local_ranked = [cols[int(i)] for i in np.asarray(order, dtype=np.int64).reshape(-1)]
            selected: list[str] = []
            seen: set[str] = set()
            for c in must_keep:
                if c not in seen:
                    selected.append(c)
                    seen.add(c)
            for c in local_ranked:
                if c in seen:
                    continue
                selected.append(c)
                seen.add(c)
                if max_k > 0 and len(selected) >= max_k:
                    break
            if len(selected) < min_k:
                for c in source_cols:
                    c_str = str(c)
                    if c_str in seen:
                        continue
                    selected.append(c_str)
                    seen.add(c_str)
                    if len(selected) >= min_k:
                        break
            ordered = [c for c in source_cols if c in set(selected)]
            return ordered if ordered else list(source_cols)

        ordered_selected = _ordered_from_score(score)

        X_fit_sel = self._align_feature_frame(X_fit, ordered_selected)
        X_eval_sel = self._align_feature_frame(X_eval, ordered_selected)
        self._save_selected_features(ordered_selected)
        regime_feature_sets: dict[str, list[str]] = {}
        if per_regime:
            rid_to_name = {0: "range", 1: "neutral", 2: "trend"}
            for rid, name in rid_to_name.items():
                r_score = regime_scores.get(rid)
                if r_score is None:
                    regime_feature_sets[name] = list(ordered_selected)
                    continue
                r_score = np.nan_to_num(np.asarray(r_score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                if r_score.size != len(cols) or float(np.nanmax(r_score)) <= 0.0:
                    regime_feature_sets[name] = list(ordered_selected)
                    continue
                regime_feature_sets[name] = _ordered_from_score(r_score)
        self._save_selected_features_by_regime(regime_feature_sets)
        self.run_summary["feature_selection"] = {
            "enabled": True,
            "selected_count": int(len(ordered_selected)),
            "total_count": int(len(source_cols)),
            "method": "l1_regime_ovr",
            "per_regime": bool(per_regime),
            "per_regime_counts": {k: int(len(v)) for k, v in regime_feature_sets.items()},
        }
        logger.info("L1 feature selection kept %s/%s columns.", len(ordered_selected), len(source_cols))
        return X_fit_sel, X_eval_sel, ordered_selected

    def _predict_model_proba(
        self,
        *,
        model: ExpertModel,
        X: Any,
        meta: Any | None = None,
    ) -> np.ndarray | None:
        try:
            kwargs = self._predict_kwargs(model, meta)
            p = self._pad_probs(model.predict_proba(X, **kwargs))
            if p.shape[0] != len(X):
                return None
            return p
        except Exception as exc:
            logger.debug("Model prediction for calibration failed: %s", exc)
            return None

    def _fit_probability_calibrators(
        self,
        X_eval: Any,
        y_eval: Any,
        meta_eval: Any | None,
    ) -> dict[str, ProbabilityCalibrator]:
        path = self.models_dir / "calibrators.joblib"
        enabled = bool(getattr(self.settings.models, "calibration_enabled", True))
        if not enabled or len(X_eval) < int(getattr(self.settings.models, "calibration_min_rows", 300) or 300):
            with contextlib.suppress(Exception):
                if path.exists():
                    path.unlink()
            return {}

        method = str(getattr(self.settings.models, "calibration_method", "platt") or "platt")
        y_arr = self._canon_labels(y_eval)
        calibrators: dict[str, ProbabilityCalibrator] = {}
        for name, model in self.models.items():
            p = self._predict_model_proba(model=model, X=X_eval, meta=meta_eval)
            if p is None:
                continue
            cal = ProbabilityCalibrator(method=method)
            if cal.fit(p, y_arr):
                calibrators[name] = cal

        try:
            if calibrators:
                joblib.dump(calibrators, path)
                logger.info("Saved %s probability calibrators (%s).", len(calibrators), method)
            elif path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to save calibrators: %s", exc)

        self.run_summary["calibration"] = {
            "enabled": bool(enabled),
            "method": method,
            "models_calibrated": int(len(calibrators)),
        }
        return calibrators

    def _build_eval_ensemble_probs(
        self,
        X_eval: Any,
        meta_eval: Any | None,
        calibrators: dict[str, ProbabilityCalibrator],
    ) -> np.ndarray | None:
        if X_eval is None or len(X_eval) == 0 or not self.models:
            return None
        probs_by_model: dict[str, np.ndarray] = {}
        meta_feature_names: list[str] = []
        meta_feature_cols: list[np.ndarray] = []
        for name, model in self.models.items():
            p = self._predict_model_proba(model=model, X=X_eval, meta=meta_eval)
            if p is None:
                continue
            cal = calibrators.get(name)
            if cal is not None:
                with contextlib.suppress(Exception):
                    p = cal.predict_proba(p)
            probs_by_model[name] = p
            meta_feature_names.append(f"{name}_buy")
            meta_feature_cols.append(np.asarray(p[:, 1], dtype=np.float32))

        if not probs_by_model:
            return None
        if self.meta_blender is not None and meta_feature_cols:
            try:
                meta_x = np.column_stack(meta_feature_cols).astype(np.float32, copy=False)
                return self._pad_probs(
                    self.meta_blender.predict_proba({"X": meta_x, "feature_names": meta_feature_names})
                )
            except Exception as exc:
                logger.debug("Meta-blender conformal source failed: %s", exc)
        stacked = np.stack(list(probs_by_model.values()), axis=0)
        return np.mean(stacked, axis=0)

    def _fit_conformal_gate(
        self,
        X_eval: Any,
        y_eval: Any,
        meta_eval: Any | None,
        calibrators: dict[str, ProbabilityCalibrator],
    ) -> dict[str, Any]:
        path = self.models_dir / "conformal_gate.json"
        enabled = bool(getattr(self.settings.risk, "conformal_enabled", True))
        if not enabled:
            with contextlib.suppress(Exception):
                if path.exists():
                    path.unlink()
            return {"enabled": False}
        probs = self._build_eval_ensemble_probs(X_eval, meta_eval, calibrators)
        if probs is None or len(probs) < 64:
            with contextlib.suppress(Exception):
                if path.exists():
                    path.unlink()
            return {"enabled": True, "fitted": False}

        gate = ConformalClassifierGate(alpha=float(getattr(self.settings.risk, "conformal_alpha", 0.10) or 0.10))
        y_arr = self._canon_labels(y_eval)
        fitted = gate.fit(probs, y_arr)
        payload = {
            "enabled": bool(enabled),
            "fitted": bool(fitted),
            "alpha": float(gate.alpha),
            "qhat": float(gate.qhat),
            "n_calib": int(gate.n_calib),
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save conformal gate: %s", exc)
        return payload

    def _resolve_cpu_budget(self) -> int:
        cpu_total = max(1, os.cpu_count() or 1)
        override = self._read_int_env("FOREX_BOT_CPU_BUDGET", None)
        if override is None:
            override = self._read_int_env("FOREX_BOT_CPU_THREADS", None)
        if override is not None and override > 0:
            return max(1, min(cpu_total, override))
        reserve = self._read_int_env("FOREX_BOT_CPU_RESERVE", 1)
        reserve = max(0, reserve or 0)
        return max(1, cpu_total - reserve)

    def _available_ram_gb(self) -> float | None:
        try:
            import psutil  # type: ignore

            return float(psutil.virtual_memory().available) / (1024**3)
        except Exception:
            return None

    def _resolve_cpu_worker_count(self, models_count: int, cpu_budget: int) -> int:
        requested = self._read_int_env("FOREX_BOT_CPU_WORKERS", None)
        if requested is not None and requested > 0:
            return max(1, min(models_count, cpu_budget, requested))

        per_worker_gb = self._read_float_env("FOREX_BOT_CPU_WORKER_GB", 2.0) or 2.0
        if per_worker_gb is not None:
            per_worker_gb = float(per_worker_gb)
        available_gb = self._available_ram_gb()
        if per_worker_gb <= 0:
            max_ram_workers = cpu_budget
        elif available_gb is None or available_gb <= 0:
            max_ram_workers = cpu_budget
        else:
            per_worker_gb = max(0.25, per_worker_gb)
            max_ram_workers = max(1, int(available_gb // per_worker_gb))

        return max(1, min(models_count, cpu_budget, max_ram_workers))

    def _resolve_gpu_worker_count(self, models_count: int) -> int:
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        if not bool(getattr(self.settings.system, "enable_gpu", False)):
            num_gpus = 0
        requested = self._read_int_env("FOREX_BOT_GPU_WORKERS", None)
        if requested is not None and requested > 0:
            num_gpus = min(num_gpus, requested)
        max_gpus = self._read_int_env("FOREX_BOT_MAX_GPUS", None)
        if max_gpus is not None and max_gpus > 0:
            num_gpus = min(num_gpus, max_gpus)
        if num_gpus <= 0:
            return 0
        return max(1, min(num_gpus, models_count))

    def _visible_gpu_ids(self, num_gpus: int) -> list[int]:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is not None:
            raw = raw.strip()
            if not raw:
                return []
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            ids: list[int] = []
            for part in parts:
                if part.isdigit():
                    ids.append(int(part))
            if ids:
                return ids[:num_gpus] if num_gpus > 0 else ids
        return list(range(max(0, num_gpus)))

    def _split_models_by_device(self, enabled_models: list[str]) -> tuple[list[str], list[str]]:
        gpu_preferred = {
            "transformer",
            "patchtst",
            "timesnet",
            "kan",
            "nbeats",
            "nbeatsx_nf",
            "tide",
            "tide_nf",
            "tabnet",
            "mlp",
            "rl_ppo",
            "rl_sac",
            "rllib_ppo",
            "rllib_sac",
            "evolution",
        }
        tree_models = {
            "lightgbm",
            "xgboost",
            "xgboost_rf",
            "xgboost_dart",
            "catboost",
            "catboost_alt",
        }

        pref = normalize_device_preference(
            getattr(self.settings.system, "enable_gpu_preference", "auto")
        )
        if pref == "cpu":
            return [], list(enabled_models)

        enable_gpu = bool(getattr(self.settings.system, "enable_gpu", False))
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        has_gpu = enable_gpu and num_gpus > 0

        tree_pref = os.environ.get("FOREX_BOT_TREE_DEVICE")
        if tree_pref is None or not str(tree_pref).strip():
            tree_pref = str(
                getattr(self.settings.models, "tree_device_preference", "")
                or ""
            ).strip()
        if not tree_pref:
            tree_pref = pref
        tree_pref = normalize_device_preference(tree_pref)
        tree_gpu = tree_pref == "gpu" or (tree_pref == "auto" and has_gpu)
        if tree_gpu:
            gpu_preferred = gpu_preferred | tree_models

        gpu_models = [m for m in enabled_models if m in gpu_preferred]
        cpu_models = [m for m in enabled_models if m not in gpu_preferred]
        return gpu_models, cpu_models

    def _parallel_models_enabled(self, enabled_models: list[str]) -> bool:
        mode = str(os.environ.get("FOREX_BOT_PARALLEL_MODELS", "auto")).lower().strip()
        if mode in {"0", "false", "no", "off"}:
            return False
        try:
            if dist.is_available() and dist.is_initialized():
                return False
        except Exception:
            pass
        if os.environ.get("FOREX_BOT_TRAIN_WORKER") == "1":
            return False
        if len(enabled_models) < 2:
            return False
        cpu_budget = self._resolve_cpu_budget()
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        if not bool(getattr(self.settings.system, "enable_gpu", False)):
            num_gpus = 0
        if mode in {"gpu", "gpus"} and num_gpus <= 0:
            return False
        if mode in {"cpu"} and cpu_budget <= 1:
            return False
        if num_gpus <= 0 and cpu_budget <= 1:
            return False
        return True

    def _write_tuned_config(self, out_path: Path) -> None:
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.settings.model_dump()  # type: ignore[attr-defined]
            out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write tuned config for workers: {exc}")

    def _export_memmap_dataset(self, X: Any, y: Any, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = list(_frame_columns(X))
        if not cols:
            try:
                n_cols = int(self._as_float32_matrix(X).shape[1])
            except Exception:
                n_cols = 0
            cols = [f"f{i}" for i in range(max(0, n_cols))]
        (out_dir / "columns.json").write_text(json.dumps(cols), encoding="utf-8")

        index_kind = "none"
        index_obj = _frame_index(X)
        try:
            if index_obj is not None and _is_datetime_index(index_obj):
                index_kind = "datetime_ns"
        except Exception:
            index_kind = "none"

        (out_dir / "meta.json").write_text(json.dumps({"index_kind": index_kind}), encoding="utf-8")

        # Persist index (optional but useful for downstream eval/debug)
        try:
            idx_ns = _index_to_int64(index_obj)
            if idx_ns is not None and idx_ns.size > 0:
                idx_mm = np.lib.format.open_memmap(
                    out_dir / "index.npy", mode="w+", dtype=np.int64, shape=(len(idx_ns),)
                )
                idx_mm[:] = idx_ns.astype(np.int64, copy=False)
                idx_mm.flush()
        except Exception:
            pass

        # Write X/y in chunks to avoid a second full-size in-memory copy.
        n_rows, n_cols = X.shape
        x_mm = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
        y_mm = np.lib.format.open_memmap(out_dir / "y.npy", mode="w+", dtype=np.int8, shape=(len(y),))

        chunk = int(os.environ.get("FOREX_BOT_MEMMAP_CHUNK_ROWS", "250000") or 250000)
        chunk = max(10_000, min(chunk, max(10_000, n_rows)))

        for start in range(0, n_rows, chunk):
            end = min(n_rows, start + chunk)
            rows = np.arange(start, end, dtype=np.int64)
            x_block = self._slice_rows(X, rows)
            y_block = self._slice_rows(y, rows)
            x_mm[start:end] = self._as_float32_matrix(x_block)
            y_mm[start:end] = np.asarray(y_block, dtype=np.int8).reshape(-1)

        x_mm.flush()
        y_mm.flush()

    @staticmethod
    def _coerce_numpy_dataset(dataset: PreparedDataset) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None] | None:
        try:
            X = ModelTrainer._as_float32_matrix(getattr(dataset, "X", None))
            y = np.asarray(getattr(dataset, "y", None), dtype=np.int8).reshape(-1)
        except Exception:
            return None
        if X.ndim != 2:
            return None
        if y.shape[0] != X.shape[0]:
            return None
        names_raw = list(getattr(dataset, "feature_names", []) or [])
        if not names_raw:
            names_raw = _frame_columns(getattr(dataset, "X", None))
        if len(names_raw) != X.shape[1]:
            names = [f"f{i}" for i in range(X.shape[1])]
        else:
            names = [str(c) for c in names_raw]

        idx = None
        idx_obj = getattr(dataset, "index", None)
        if idx_obj is None:
            idx_obj = _frame_index(getattr(dataset, "X", None))
        idx_ns = _index_to_int64(idx_obj)
        if idx_ns is not None and idx_ns.shape[0] == X.shape[0]:
            idx = idx_ns
        return X, y, names, idx

    def _export_memmap_dataset_numpy(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        index_ns: np.ndarray | None,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "columns.json").write_text(json.dumps([str(c) for c in feature_names]), encoding="utf-8")
        index_kind = "datetime_ns" if index_ns is not None else "none"
        (out_dir / "meta.json").write_text(json.dumps({"index_kind": index_kind}), encoding="utf-8")

        n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
        x_mm = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
        y_mm = np.lib.format.open_memmap(out_dir / "y.npy", mode="w+", dtype=np.int8, shape=(n_rows,))

        chunk = int(os.environ.get("FOREX_BOT_MEMMAP_CHUNK_ROWS", "250000") or 250000)
        chunk = max(10_000, min(chunk, max(10_000, n_rows)))
        for start in range(0, n_rows, chunk):
            end = min(n_rows, start + chunk)
            x_mm[start:end] = X[start:end]
            y_mm[start:end] = y[start:end]
        x_mm.flush()
        y_mm.flush()

        if index_ns is not None:
            idx_mm = np.lib.format.open_memmap(out_dir / "index.npy", mode="w+", dtype=np.int64, shape=(n_rows,))
            idx_mm[:] = index_ns.astype(np.int64, copy=False)
            idx_mm.flush()

    def _materialize_numpy_memmap_dataset(self, dataset: PreparedDataset) -> Path | None:
        coerced = self._coerce_numpy_dataset(dataset)
        if coerced is None:
            return None
        X, y, names, idx = coerced
        cache_root = Path(getattr(self.settings.system, "cache_dir", "cache")) / "pandas_free_pool"
        run_id = f"{int(time.time())}_{os.getpid()}"
        out_dir = cache_root / f"pool_{run_id}"
        try:
            self._export_memmap_dataset_numpy(
                X=X,
                y=y,
                feature_names=names,
                index_ns=idx,
                out_dir=out_dir,
            )
            return out_dir
        except Exception as exc:
            logger.warning("Failed to materialize numpy memmap dataset: %s", exc, exc_info=True)
            with contextlib.suppress(Exception):
                shutil.rmtree(out_dir, ignore_errors=True)
            return None

    def _schedule_models(self, models: list[str], workers: int) -> list[list[str]]:
        # Rough weights to reduce wall-time skew (heavy models first).
        base = dict(getattr(self.benchmarker, "COMPLEXITY_MAP", {}) or {})
        base.setdefault("genetic", 0.002)
        base.setdefault("rllib_ppo", base.get("rl_ppo", 0.005))
        base.setdefault("rllib_sac", base.get("rl_sac", 0.008))

        weights = {m: float(base.get(m, 0.001)) for m in models}
        ordered = sorted(models, key=lambda m: weights.get(m, 0.0), reverse=True)

        buckets: list[list[str]] = [[] for _ in range(workers)]
        loads: list[float] = [0.0 for _ in range(workers)]
        for m in ordered:
            i = int(min(range(workers), key=lambda j: loads[j]))
            buckets[i].append(m)
            loads[i] += weights.get(m, 0.0)
        return buckets

    def _merge_worker_dirs(self, worker_dirs: list[Path]) -> None:
        skip = {
            "active_models.pkl",
            "meta_blender.joblib",
            "run_summary.json",
            "incremental_stats.json",
            "global_incremental_progress.json",
            "worker_manifest.json",
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for wdir in worker_dirs:
            if not wdir.exists():
                continue
            for item in wdir.iterdir():
                if item.name in skip:
                    continue
                dest = self.models_dir / item.name
                try:
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                except Exception as exc:
                    logger.warning(f"Failed to merge worker artifact {item}: {exc}")

    def _train_models_parallel(
        self,
        enabled_models: list[str],
        X_fit: Any,
        y_fit: Any,
        *,
        meta_fit: Any | None = None,
        stop_event: asyncio.Event | None,
        memmap_dataset_dir: Path | str | None = None,
    ) -> dict[str, float]:
        """
        Train models across available CPUs/GPUs using subprocess workers pinned via CUDA_VISIBLE_DEVICES.
        Returns per-model durations (best effort).
        """
        mode = str(os.environ.get("FOREX_BOT_PARALLEL_MODELS", "auto")).lower().strip()
        cpu_budget = self._resolve_cpu_budget()

        enable_gpu = bool(getattr(self.settings.system, "enable_gpu", False))
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        if not enable_gpu:
            num_gpus = 0

        gpu_models, cpu_models = self._split_models_by_device(enabled_models)
        if num_gpus <= 0:
            gpu_models = []
            cpu_models = list(enabled_models)

        if mode in {"gpu", "gpus"}:
            cpu_models = []
        elif mode in {"cpu"}:
            gpu_models = []
            cpu_models = list(enabled_models)

        if meta_fit is None:
            require_meta = {"genetic", "rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac"}
            dropped = [m for m in (gpu_models + cpu_models) if m in require_meta]
            if dropped:
                logger.info(f"Parallel: skipping metadata-dependent models without metadata: {dropped}")
            gpu_models = [m for m in gpu_models if m not in require_meta]
            cpu_models = [m for m in cpu_models if m not in require_meta]

        visible_gpu_ids: list[int] = []
        if num_gpus > 0 and gpu_models:
            visible_gpu_ids = self._visible_gpu_ids(num_gpus)
            if not visible_gpu_ids:
                logger.warning("Parallel: GPUs requested but none are visible; falling back to CPU workers.")
                cpu_models = list(cpu_models) + list(gpu_models)
                gpu_models = []

        gpu_workers = self._resolve_gpu_worker_count(len(gpu_models)) if gpu_models else 0
        if gpu_workers > 0 and visible_gpu_ids:
            gpu_workers = min(gpu_workers, len(visible_gpu_ids))
        cpu_workers = self._resolve_cpu_worker_count(len(cpu_models), cpu_budget) if cpu_models else 0

        if gpu_workers <= 0 and cpu_workers <= 0:
            raise RuntimeError("Parallel training requested but no workers available")

        gpu_specs: list[dict[str, Any]] = []
        cpu_specs: list[dict[str, Any]] = []

        if gpu_workers > 0 and gpu_models:
            # Dynamic queue: one model per spec; GPUs pull next when free.
            for idx, model in enumerate(gpu_models):
                gpu_specs.append(
                    {
                        "kind": "gpu",
                        "label": f"gpu_{idx}_{model}",
                        "gpu_id": None,
                        "models": [model],
                    }
                )

        if cpu_workers > 0 and cpu_models:
            buckets = self._schedule_models(cpu_models, cpu_workers)
            for wid, models in enumerate(buckets):
                if not models:
                    continue
                cpu_specs.append(
                    {
                        "kind": "cpu",
                        "label": f"cpu_{wid}",
                        "gpu_id": None,
                        "models": models,
                    }
                )

        total_concurrent = (gpu_workers if gpu_specs else 0) + (cpu_workers if cpu_specs else 0)
        if total_concurrent <= 0:
            raise RuntimeError("Parallel training requested but no workers were scheduled")

        threads_per_worker = max(1, cpu_budget // max(1, total_concurrent))
        for spec in gpu_specs + cpu_specs:
            spec["threads"] = threads_per_worker

        cache_root = Path(getattr(self.settings.system, "cache_dir", "cache")) / "parallel_training"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = cache_root / f"run_{run_id}_{os.getpid()}"
        dataset_dir = Path(memmap_dataset_dir) if memmap_dataset_dir else (run_dir / "dataset")
        cfg_path = run_dir / "tuned_config.yaml"
        logs_dir = self.logs_dir / "parallel_workers" / run_dir.name
        workers_root = self.models_dir / "_workers" / run_dir.name

        logs_dir.mkdir(parents=True, exist_ok=True)
        workers_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Parallel model training enabled: gpu_workers=%s cpu_workers=%s cpu_budget=%s threads/worker=%s",
            gpu_workers,
            cpu_workers,
            cpu_budget,
            threads_per_worker,
        )

        self._write_tuned_config(cfg_path)
        cfg_exists = cfg_path.exists()

        dataset_ok = False
        if memmap_dataset_dir:
            try:
                dataset_ok = (
                    dataset_dir.exists()
                    and (dataset_dir / "X.npy").exists()
                    and (dataset_dir / "y.npy").exists()
                    and (dataset_dir / "columns.json").exists()
                )
            except Exception:
                dataset_ok = False

        if dataset_ok:
            logger.info(f"Parallel: Reusing memmap dataset at {dataset_dir}.")
        else:
            if memmap_dataset_dir:
                logger.warning(
                    f"Parallel: Provided memmap dataset missing or incomplete; regenerating at {dataset_dir}."
                )
            self._export_memmap_dataset(X_fit, y_fit, dataset_dir)

        metadata_path = None
        if meta_fit is not None:
            try:
                metadata_path = self._persist_metadata_artifact(meta_fit, run_dir / "metadata.pkl")
            except Exception as exc:
                metadata_path = None
                logger.warning(f"Failed to persist metadata for workers: {exc}")

        entry = _find_entrypoint_script()

        cpu_max_concurrent = self._read_int_env("FOREX_BOT_CPU_MAX_CONCURRENT_MODELS", None)

        # Smart RAM-aware concurrency calculation
        if cpu_max_concurrent is None or cpu_max_concurrent <= 0:
            try:
                import psutil
                from ..core.system import resolve_cpu_budget

                available_gb = psutil.virtual_memory().available / 1e9
                cpu_budget = resolve_cpu_budget()

                # Estimate: each concurrent model needs ~3GB peak during training
                ram_limited = max(1, int(available_gb / 3.0))
                # CPU-limited: minimum 2 threads per model for efficiency
                cpu_limited = max(1, cpu_budget // 2)

                # Take the minimum to respect both constraints
                cpu_max_concurrent = min(ram_limited, cpu_limited)
                logger.info(f"Auto-calculated cpu_max_concurrent={cpu_max_concurrent} "
                           f"(RAM-limited: {ram_limited}, CPU-limited: {cpu_limited}, "
                           f"available: {available_gb:.1f}GB, cores: {cpu_budget})")
            except Exception as e:
                logger.debug(f"Failed to auto-calculate concurrency: {e}")
                cpu_max_concurrent = 0  # Fall back to default

        procs: list[tuple[str, Path, subprocess.Popen, Any]] = []
        failed: list[tuple[str, int]] = []
        running: list[dict[str, Any]] = []

        pending_gpu = list(gpu_specs)
        pending_cpu = list(cpu_specs)
        available_gpu_ids = list(visible_gpu_ids[:gpu_workers]) if gpu_workers > 0 else []

        def _launch_worker(spec: dict[str, Any], gpu_id: int | None = None) -> None:
            models = spec.get("models") or []
            if not models:
                return
            label = str(spec.get("label") or len(procs))
            out_dir = workers_root / f"worker_{label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"worker_{label}.log"

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FOREX_BOT_TRAIN_WORKER"] = "1"
            if cfg_exists:
                env["CONFIG_FILE"] = str(cfg_path)

            if spec.get("kind") == "gpu":
                env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else str(gpu_id)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            worker_threads = int(spec.get("threads") or threads_per_worker)
            worker_threads = max(1, worker_threads)
            env["FOREX_BOT_CPU_BUDGET"] = str(worker_threads)
            env["FOREX_BOT_CPU_THREADS"] = str(worker_threads)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[k] = str(worker_threads)

            if spec.get("kind") == "gpu":
                max_concurrent = 1
            else:
                if cpu_max_concurrent is not None and cpu_max_concurrent > 0:
                    max_concurrent = min(len(models), int(cpu_max_concurrent))
                else:
                    max_concurrent = 0

            cmd = [
                sys.executable,
                str(entry),
                "--_worker",
                "--dataset-dir",
                str(dataset_dir),
                "--models",
                ",".join(models),
                "--out-dir",
                str(out_dir),
                "--cpu-threads",
                str(worker_threads),
                "--max-concurrent-models",
                str(max_concurrent),
            ]
            if metadata_path is not None and metadata_path.exists():
                cmd += ["--metadata-path", str(metadata_path)]

            log_f = open(log_path, "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(entry.parent),
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                try:
                    log_f.close()
                except Exception:
                    pass
                raise

            procs.append((label, out_dir, proc, log_f))
            running.append(
                {
                    "label": label,
                    "out_dir": out_dir,
                    "proc": proc,
                    "log_f": log_f,
                    "kind": spec.get("kind"),
                    "gpu_id": gpu_id,
                }
            )
            logger.info(
                "Worker %s (%s): models=%s threads=%s log=%s",
                label,
                spec.get("kind"),
                models,
                worker_threads,
                log_path,
            )

        try:
            for spec in pending_cpu:
                _launch_worker(spec)
            pending_cpu = []

            while pending_gpu and available_gpu_ids:
                _launch_worker(pending_gpu.pop(0), available_gpu_ids.pop(0))

            while running or pending_gpu:
                if stop_event and stop_event.is_set():
                    raise KeyboardInterrupt("Stop requested")

                for entry in list(running):
                    proc = entry.get("proc")
                    if proc is None or proc.poll() is None:
                        continue
                    running.remove(entry)
                    if entry.get("kind") == "gpu" and entry.get("gpu_id") is not None:
                        available_gpu_ids.append(int(entry.get("gpu_id")))
                    try:
                        if proc.returncode not in (None, 0):
                            failed.append((str(entry.get("label")), int(proc.returncode)))
                    except Exception:
                        failed.append((str(entry.get("label")), 1))

                while pending_gpu and available_gpu_ids:
                    _launch_worker(pending_gpu.pop(0), available_gpu_ids.pop(0))

                if not pending_gpu and not running:
                    break
                time.sleep(2)

            if failed:
                logger.error(f"Parallel workers failed: {failed}. See logs under {logs_dir}")
                raise RuntimeError(f"Parallel workers failed: {failed}")
        except Exception:
            for _, _, p, _ in procs:
                try:
                    if p.poll() is None:
                        p.terminate()
                except Exception:
                    pass
            raise
        finally:
            for _, _, _, lf in procs:
                try:
                    lf.close()
                except Exception:
                    pass

        worker_dirs = [out for _, out, _, _ in procs]
        self._merge_worker_dirs(worker_dirs)

        # Aggregate durations from worker manifests.
        durations: dict[str, float] = {}
        trained_models: list[str] = []
        for wdir in worker_dirs:
            try:
                manifest = json.loads((wdir / "worker_manifest.json").read_text(encoding="utf-8"))
                t = manifest.get("trained", [])
                if isinstance(t, list):
                    trained_models.extend([str(x) for x in t if str(x)])
                d = manifest.get("durations_sec", {})
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            durations[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception:
                continue

        # Load full model set into this trainer instance for downstream stacking/eval.
        try:
            import joblib

            if trained_models:
                trained_set = set(trained_models)
                ordered = [m for m in enabled_models if m in trained_set]
            else:
                ordered = list(enabled_models)
            joblib.dump(ordered, self.models_dir / "active_models.pkl")
        except Exception:
            pass

        if self._pandas_free_enabled():
            self.models = {}
            self.meta_blender = None
        else:
            try:
                self.models, self.meta_blender = self.persistence.load_models()
            except Exception as exc:
                logger.warning(f"Failed to load merged models after parallel training: {exc}", exc_info=True)

        return durations

    def _train_all_pandas_free_memmap(
        self,
        *,
        optimize: bool,
        stop_event: asyncio.Event | None,
        models_override: list[str] | None,
        exclude_models: list[str] | None,
        memmap_dataset_dir: Path | str | None,
    ) -> bool:
        if not self._memmap_dataset_complete(memmap_dataset_dir):
            return False

        dataset_dir = Path(memmap_dataset_dir)
        if optimize:
            logger.info("Pandas-free mode: disabling HPO optimization for stable rust-tree training path.")

        enabled_models = self._get_enabled_models()
        enabled_models = self._maybe_shard_models(enabled_models)
        if models_override is not None:
            override_ordered = [m for m in models_override if m in enabled_models]
            if not override_ordered:
                logger.warning("No override models available; skipping training.")
                return True
            enabled_models = override_ordered
        if exclude_models:
            exclude_set = set(exclude_models)
            enabled_models = [m for m in enabled_models if m not in exclude_set]
            if not enabled_models:
                logger.warning("All models excluded; skipping training.")
                return True
        if not enabled_models:
            logger.warning("Pandas-free memmap: no compatible Rust-backed models selected; skipping training.")
            self.run_summary["pandas_free_memmap"] = {
                "enabled": True,
                "dataset_dir": str(dataset_dir),
                "models": [],
                "reason": "no_compatible_rust_models",
            }
            if not self.distributed_enabled or self.rank == 0:
                self.persistence.save_run_summary(self.run_summary)
                self.persistence.save_active_models_list(self.models)
                with contextlib.suppress(Exception):
                    self.persistence.save_models_bundle(self.models)
            if self.distributed_enabled:
                with contextlib.suppress(Exception):
                    dist.barrier()
            self.persistence.cleanup_logs()
            return True

        feature_columns: list[str] = []
        with contextlib.suppress(Exception):
            loaded = json.loads((dataset_dir / "columns.json").read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                feature_columns = [str(c) for c in loaded]

        logger.info(
            "Pandas-free memmap training path enabled: models=%s rows_source=%s",
            len(enabled_models),
            dataset_dir,
        )

        durations = self._train_models_parallel(
            enabled_models,
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
            meta_fit=None,
            stop_event=stop_event,
            memmap_dataset_dir=dataset_dir,
        )

        train_samples = 0
        with contextlib.suppress(Exception):
            x_mm = np.load(dataset_dir / "X.npy", mmap_mode="r")
            train_samples = int(x_mm.shape[0]) if getattr(x_mm, "ndim", 0) >= 1 else 0

        self.run_summary["feature_columns"] = feature_columns
        self.run_summary["train_durations_sec"] = durations
        self.run_summary["train_samples"] = int(train_samples)
        self.run_summary["train_hardware"] = {
            "enable_gpu": bool(getattr(self.settings.system, "enable_gpu", False)),
            "num_gpus": int(getattr(self.settings.system, "num_gpus", 1)),
            "device": str(getattr(self.settings.system, "device", "cpu")),
        }
        self.run_summary["pandas_free_memmap"] = {
            "enabled": True,
            "dataset_dir": str(dataset_dir),
            "models": list(enabled_models),
        }

        if not self.distributed_enabled or self.rank == 0:
            self.persistence.save_run_summary(self.run_summary)
            self.persistence.save_active_models_list(enabled_models)
            with contextlib.suppress(Exception):
                self.persistence.save_models_bundle(enabled_models)
        if self.distributed_enabled:
            with contextlib.suppress(Exception):
                dist.barrier()
        self.persistence.cleanup_logs()
        logger.info("Training Complete.")
        return True

    def train_all(
        self,
        dataset: PreparedDataset,
        optimize: bool = True,
        stop_event: asyncio.Event | None = None,
        models_override: list[str] | None = None,
        exclude_models: list[str] | None = None,
        memmap_dataset_dir: Path | str | None = None,
    ) -> None:
        logger.info("Starting training cycle...")
        pandas_free = self._pandas_free_enabled()
        strict_pandas_free = self._pandas_free_strict_enabled()
        if pandas_free and optimize:
            logger.info("Pandas-free mode: disabling HPO optimization for stable rust-tree training path.")
            optimize = False
        pandas_free_memmap_dir: Path | str | None = memmap_dataset_dir
        if pandas_free and not self._memmap_dataset_complete(pandas_free_memmap_dir):
            pandas_free_memmap_dir = self._materialize_numpy_memmap_dataset(dataset)
        if pandas_free and not self._memmap_dataset_complete(pandas_free_memmap_dir):
            msg = (
                "Pandas-free mode: memmap dataset is unavailable; "
                "aborting instead of legacy tabular fallback."
            )
            if strict_pandas_free:
                logger.error("%s Set FOREX_BOT_PANDAS_FREE_STRICT=0 to allow fallback.", msg)
                return
            logger.error("%s (strict off) — legacy tabular fallback path removed; please regenerate memmap dataset.", msg)
            return
        if pandas_free and self._train_all_pandas_free_memmap(
            optimize=optimize,
            stop_event=stop_event,
            models_override=models_override,
            exclude_models=exclude_models,
            memmap_dataset_dir=pandas_free_memmap_dir,
        ):
            return
        if pandas_free and strict_pandas_free:
            logger.error(
                "Pandas-free strict mode: memmap training path did not complete; "
                "aborting instead of using legacy tabular fallback."
            )
            return
        # Enforce GPU-only mode when explicitly requested (or preference is GPU)
        try:
            gpu_only_env = str(os.environ.get("FOREX_BOT_GPU_ONLY", "")).strip().lower()
            gpu_only = gpu_only_env in {"1", "true", "yes", "on"}
        except Exception:
            gpu_only = False
        if not gpu_only:
            pref = str(getattr(self.settings.system, "enable_gpu_preference", "auto")).strip().lower()
            if pref == "gpu":
                gpu_only = True
                os.environ["FOREX_BOT_GPU_ONLY"] = "1"
        if gpu_only:
            os.environ.setdefault("FOREX_BOT_TREE_DEVICE", "gpu")

        # 1. Setup Logging
        writer = None
        if TENSORBOARD_AVAILABLE:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(log_dir=str(self.logs_dir / f"training_{run_id}"))

        # 2. Prepare Data (META-LABELING FILTER)
        # Only train on rows where a base strategy signal exists.
        # This prevents the model from learning to just predict "No Trade" (accuracy padding).
        X_raw = dataset.X
        y_raw = dataset.y if _is_series(dataset.y) else np.asarray(dataset.y)

        use_filter = bool(getattr(self.settings.models, "filter_to_base_signal", True))
        has_base_signal = _frame_has_column(X_raw, "base_signal")
        if use_filter and has_base_signal:
            base_signal = _frame_extract_column(X_raw, "base_signal")
            if base_signal is None:
                mask = np.ones(int(len(X_raw)), dtype=bool)
            else:
                mask = np.asarray(base_signal).reshape(-1) != 0
            active_rows = int(np.count_nonzero(mask))
            total_rows = int(len(X_raw))
            coverage = float(active_rows) / float(max(1, total_rows))

            try:
                min_rows = int(os.environ.get("FOREX_BOT_BASE_SIGNAL_MIN_ROWS", "100") or 100)
            except Exception:
                min_rows = 100
            min_rows = max(0, min_rows)

            try:
                min_cov = float(
                    os.environ.get("FOREX_BOT_BASE_SIGNAL_MIN_COVERAGE", "0.0") or 0.0
                )
            except Exception:
                min_cov = 0.0
            min_cov = max(0.0, min(1.0, min_cov))

            logger.info(
                "Meta-Labeling: base_signal coverage=%.3f%% (%s/%s rows).",
                coverage * 100.0,
                active_rows,
                total_rows,
            )

            if active_rows < min_rows or (min_cov > 0.0 and coverage < min_cov):
                logger.warning("Insufficient base signals for meta-labeling. Training on all data (fallback).")
                X = X_raw
                y = y_raw
                meta_subset = dataset.metadata
            elif active_rows == total_rows:
                # Avoid an unnecessary full-frame copy when already pre-filtered upstream.
                logger.info(f"Meta-Labeling: Using pre-filtered dataset ({len(X_raw)} active signal events).")
                X = X_raw
                y = y_raw
                meta_subset = dataset.metadata
            else:
                logger.info(
                    f"Meta-Labeling: Filtering to {active_rows} active signal events (from {total_rows} rows)."
                )
                mask_arr = np.asarray(mask)
                X = self._slice_rows(X_raw, mask_arr)
                y = self._slice_rows(y_raw, mask_arr)
                meta_subset = self._slice_rows(dataset.metadata, mask_arr) if dataset.metadata is not None else None
        else:
            if not use_filter:
                logger.info("Meta-Labeling: base_signal filter disabled; using full dataset.")
            elif not has_base_signal:
                logger.warning("Meta-Labeling: base_signal column missing; using full dataset.")
            X = X_raw
            y = y_raw
            meta_subset = dataset.metadata

        try:
            self.run_summary["feature_columns"] = list(_frame_columns(X))
        except Exception:
            self.run_summary["feature_columns"] = []

        # Holdout split for out-of-sample evaluation + stacking.
        #
        # IMPORTANT: For pooled multi-symbol datasets, row-order is often NOT time-ordered
        # (e.g., concatenated by symbol). A positional split would leak "future" rows into
        # the fit set and invalidate evaluation. Prefer a timestamp-based split when possible.
        x_index = _frame_index(X)
        order = _sorted_time_order(x_index, len(X)) if _is_datetime_index(x_index) else None
        if order is not None:
            X = self._slice_rows(X, order)
            y = self._slice_rows(y, order)
            if meta_subset is not None:
                meta_subset = self._slice_rows(meta_subset, order)
        n = len(y)
        holdout_pct = float(getattr(self.settings.models, "train_holdout_pct", 0.2) or 0.2)
        holdout_pct = float(min(0.5, max(0.0, holdout_pct)))
        min_split = 50  # Lower threshold since we filtered

        X_fit = X
        y_fit = y
        X_eval = self._slice_rows(X, np.arange(0, dtype=np.int64))
        y_eval = self._slice_rows(y, np.arange(0, dtype=np.int64))
        meta_fit = None
        meta_eval = None
        holdout_cutoff: str | None = None
        split_strategy = "none"

        def _finalize_summary() -> None:
            fit_n = int(len(X_fit))
            eval_n = int(len(X_eval))
            self.run_summary["train_holdout_pct"] = float(holdout_pct if eval_n > 0 else 0.0)
            self.run_summary["train_holdout_n"] = int(eval_n)
            self.run_summary["train_fit_n"] = int(fit_n)
            self.run_summary["train_holdout_split"] = {
                "strategy": split_strategy,
                "cutoff": holdout_cutoff,
            }

        # Time-based split if index is datetime (works even if the frame is not pre-sorted).
        x_index = _frame_index(X)
        if holdout_pct > 0.0 and _is_datetime_index(x_index) and n > 0:
            try:
                idx_ns = _index_to_int64(x_index)
                if idx_ns is None or idx_ns.size != int(n):
                    raise ValueError("index conversion failed")
                nat = np.iinfo(np.int64).min
                valid = idx_ns != nat

                if valid.any():
                    times_valid = np.array(idx_ns[valid], copy=True)
                    fit_frac = float(1.0 - holdout_pct)
                    k = int(max(0, min(len(times_valid) - 1, int(len(times_valid) * fit_frac) - 1)))
                    cut_ns = int(np.partition(times_valid, k)[k])
                    cutoff_ts = self._utc_iso_from_ns(cut_ns)

                    fit_mask = idx_ns <= cut_ns
                    eval_mask = idx_ns > cut_ns
                    fit_n = int(fit_mask.sum())
                    eval_n = int(eval_mask.sum())

                    if fit_n >= min_split and eval_n >= min_split:
                        X_fit = self._slice_rows(X, fit_mask)
                        y_fit = self._slice_rows(y, fit_mask)
                        X_eval = self._slice_rows(X, eval_mask)
                        y_eval = self._slice_rows(y, eval_mask)
                        if meta_subset is not None:
                            with contextlib.suppress(Exception):
                                meta_fit = self._slice_rows(meta_subset, fit_mask)
                            with contextlib.suppress(Exception):
                                meta_eval = self._slice_rows(meta_subset, eval_mask)
                        holdout_cutoff = cutoff_ts
                        split_strategy = "time_cutoff"
                        _finalize_summary()
                    else:
                        split_strategy = "time_cutoff_insufficient"
            except Exception as exc:
                logger.warning(f"Holdout time-based split failed; falling back to positional: {exc}")

        # Fallback: positional split (assumes row-order is time-ordered).
        if split_strategy in {"none", "time_cutoff_insufficient"}:
            holdout_n = int(n * holdout_pct)
            eval_start = max(0, n - holdout_n)
            if eval_start < min_split or (n - eval_start) < min_split:
                eval_start = n  # disable holdout

            train_rows = np.arange(eval_start, dtype=np.int64)
            eval_rows = np.arange(eval_start, n, dtype=np.int64)
            X_fit = self._slice_rows(X, train_rows)
            y_fit = self._slice_rows(y, train_rows)
            X_eval = self._slice_rows(X, eval_rows)
            y_eval = self._slice_rows(y, eval_rows)

            meta_fit = None
            meta_eval = None
            if meta_subset is not None:
                with contextlib.suppress(Exception):
                    meta_fit = self._slice_rows(meta_subset, train_rows)
                with contextlib.suppress(Exception):
                    meta_eval = self._slice_rows(meta_subset, eval_rows)

            split_strategy = "positional" if eval_start < n else "positional_disabled"
            _finalize_summary()

        selected_features: list[str] = []
        try:
            X_fit, X_eval, selected_features = self._apply_l1_feature_selection(X_fit, y_fit, X_eval)
            if _frame_columns(X):
                X = self._align_feature_frame(X, selected_features)
            self.run_summary["feature_columns"] = list(selected_features)
        except Exception as exc:
            logger.warning("L1 feature selection failed; continuing with original features: %s", exc)
            selected_features = list(_frame_columns(X_fit))
            self._save_selected_features(selected_features)
            self._save_selected_features_by_regime({})

        # Feature drift detection between train and eval sets
        if len(X_fit) > 0 and len(X_eval) > 0:
            try:
                drift_result = detect_feature_drift(X_fit, X_eval, threshold=0.1)
                self.run_summary["feature_drift"] = {
                    "summary": drift_result["summary"],
                    "critical": drift_result["critical"],
                    "drifted_count": len(drift_result["drifted_features"]),
                    "top_drifted": drift_result["drifted_features"][:10],
                }
                if drift_result["critical"]:
                    logger.warning(
                        f"Critical feature drift detected! {drift_result['summary']}. "
                        "Model predictions may be unreliable on recent data."
                    )
            except Exception as e:
                logger.debug(f"Feature drift detection failed: {e}")

        if len(X_fit) > 200 and len(X_eval) > 200 and bool(
            getattr(self.settings.models, "adversarial_validation_enabled", True)
        ):
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import train_test_split

                max_rows = int(getattr(self.settings.models, "adversarial_validation_max_rows", 200_000) or 200_000)
                n_fit = min(len(X_fit), max(100, max_rows // 2))
                n_eval = min(len(X_eval), max(100, max_rows // 2))
                x_fit_adv = self._sample_rows(X_fit, n_fit, random_state=42)
                x_eval_adv = self._sample_rows(X_eval, n_eval, random_state=42)
                x_fit_np = self._as_float32_matrix(x_fit_adv)
                x_eval_np = self._as_float32_matrix(x_eval_adv)
                x_adv = np.concatenate([x_fit_np, x_eval_np], axis=0)
                y_adv = np.concatenate(
                    [
                        np.zeros(len(x_fit_adv), dtype=np.int8),
                        np.ones(len(x_eval_adv), dtype=np.int8),
                    ]
                )
                x_tr, x_te, y_tr, y_te = train_test_split(
                    x_adv,
                    y_adv,
                    test_size=0.30,
                    random_state=42,
                    stratify=y_adv,
                )
                clf = LogisticRegression(max_iter=300)
                clf.fit(x_tr, y_tr)
                adv_acc = float(clf.score(x_te, y_te))
                adv_thr = float(getattr(self.settings.models, "adversarial_validation_alert_acc", 0.55) or 0.55)
                self.run_summary["adversarial_validation"] = {
                    "accuracy": adv_acc,
                    "threshold": adv_thr,
                    "rows_train": int(len(x_fit_adv)),
                    "rows_eval": int(len(x_eval_adv)),
                    "non_stationary": bool(adv_acc > adv_thr),
                }
                if adv_acc > adv_thr:
                    logger.warning(
                        "Adversarial validation flagged train/eval shift (acc=%.3f > %.3f).",
                        adv_acc,
                        adv_thr,
                    )
            except Exception as exc:
                logger.debug(f"Adversarial validation skipped: {exc}")

        # Basic Cast
        try:
            X = self._coerce_feature_container_float32(X)
        except Exception as e:
            logger.debug(f"Float32 casting failed: {e}")

        # 3. Optimization
        best_params = {}
        if optimize and getattr(self.optimizer, "available", False):
            best_params = self.optimizer.optimize_all(X_fit, y_fit, meta_fit, stop_event=stop_event)
        else:
            best_params = self.optimizer.load_params()

        # Preserve metadata even for multi-symbol runs so OHLC-dependent models can train globally.
        # NOTE: downstream models that assume a single continuous series should handle the symbol column explicitly.
        meta_fit_models = meta_fit
        meta_eval_models = meta_eval
        if meta_fit is not None and _frame_has_column(meta_fit, "symbol"):
            logger.info("Multi-symbol metadata detected; passing through to OHLC-dependent models for global training.")

        # 4. Model Selection
        enabled_models = self._get_enabled_models()
        enabled_models = self._maybe_shard_models(enabled_models)
        if models_override is not None:
            override_ordered = [m for m in models_override if m in enabled_models]
            if not override_ordered:
                logger.warning("No override models available; skipping training.")
                return
            enabled_models = override_ordered
        if exclude_models:
            exclude_set = set(exclude_models)
            enabled_models = [m for m in enabled_models if m not in exclude_set]
            if not enabled_models:
                logger.warning("All models excluded; skipping training.")
                return

        # 5. Benchmark
        device = (
            "cuda"
            if bool(getattr(self.settings.system, "enable_gpu", False))
            else str(getattr(self.settings.system, "device", "cpu"))
        )
        bench_result = self.benchmarker.run_micro_benchmark(X_fit, y_fit, device)
        est_time = self.benchmarker.estimate_time(
            enabled_models,
            len(X_fit),
            bench_result,
            bool(getattr(self.settings.system, "enable_gpu", False)),
            int(getattr(self.settings.system, "num_gpus", 1)),
            context="full",
            historical_durations=self._historical_durations,
            historical_n=self._historical_samples,
            historical_gpu=(
                self._historical_hardware.get("enable_gpu"),
                self._historical_hardware.get("num_gpus", 1),
            ),
            incremental_stats=self.incremental_stats,
            probe_kwargs={
                "X": self._slice_rows(X_fit, np.arange(min(10_000, len(X_fit)), dtype=np.int64)),
                "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                "device": device,
                "steps": 8,
            },
        )
        logger.info(
            f"Estimated Training Time (ALL {len(enabled_models)} models): {est_time / 3600:.1f} hours "
            f"({est_time / 60:.0f} minutes)"
        )
        if len(enabled_models) > 0:
            logger.info(f"  → Average per model: {est_time / len(enabled_models) / 3600:.2f} hours")

        # 6. Train Loop
        durations = {}
        total = len(enabled_models)

        trained_in_parallel = False
        if self._parallel_models_enabled(enabled_models):
            try:
                durations = self._train_models_parallel(
                    enabled_models,
                    X_fit,
                    y_fit,
                    meta_fit=meta_fit_models,
                    stop_event=stop_event,
                    memmap_dataset_dir=memmap_dataset_dir,
                )
                trained_in_parallel = True
            except Exception as exc:
                logger.warning(f"Parallel model training failed; falling back to sequential: {exc}", exc_info=True)
                durations = {}
                trained_in_parallel = False

        # Device plan (round-robin via factory), just for logging
        if (
            not trained_in_parallel
            and bool(getattr(self.settings.system, "enable_gpu", False))
            and getattr(self.settings.system, "num_gpus", 1) > 1
        ):
            mapping = {}
            for idx, name in enumerate(enabled_models, 1):
                if self.factory.available_gpus:
                    mapping[name] = self.factory.available_gpus[(idx - 1) % len(self.factory.available_gpus)]
            if mapping:
                logger.info(f"GPU assignment: {mapping}")

        for idx, name in enumerate(enabled_models, 1):
            if trained_in_parallel:
                break
            if stop_event and stop_event.is_set():
                break

            # Some experts require OHLC metadata (e.g., GA TA-Lib strategies, prop-firm RL env).
            # In pooled multi-symbol runs, we intentionally disable metadata to avoid single-series leakage.
            # Skip those experts up-front instead of calling fit() with missing metadata,
            # which would otherwise produce noisy warnings.
            # Allow OHLC-dependent models to run with global (multi-symbol) metadata; they should
            # handle the 'symbol' column internally to avoid leakage.
            if name in {"genetic", "rl_ppo", "rl_sac"} and meta_fit_models is None:
                logger.info(f"[{idx}/{total}] Skipping {name}: OHLC metadata unavailable for this training run.")
                durations[name] = 0.0
                continue

            t0 = time.perf_counter()
            try:
                logger.info(f"[{idx}/{total}] Training {name}...")

                # Ensure time-ordered data for models that rely on chronological sequences
                time_order_models = {"genetic", "rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac", "evolution"}
                fit_index = _frame_index(X_fit)
                if name in time_order_models and _is_datetime_index(fit_index):
                    fit_order = _sorted_time_order(fit_index, len(X_fit))
                    if fit_order is not None:
                        X_fit = self._slice_rows(X_fit, fit_order)
                        y_fit = self._slice_rows(y_fit, fit_order)
                        if meta_fit_models is not None:
                            meta_fit_models = self._slice_rows(meta_fit_models, fit_order)
                    eval_index = _frame_index(X_eval)
                    if len(X_eval) > 0 and _is_datetime_index(eval_index):
                        eval_order = _sorted_time_order(eval_index, len(X_eval))
                        if eval_order is not None:
                            X_eval = self._slice_rows(X_eval, eval_order)
                            y_eval = self._slice_rows(y_eval, eval_order)
                            if meta_eval_models is not None:
                                meta_eval_models = self._slice_rows(meta_eval_models, eval_order)

                model = self.factory.create_model(name, best_params, idx)

                # Optional FSDP wrapping for torch models when multiple GPUs & USE_FSDP=1
                env_fsdp = os.environ.get("USE_FSDP", "0").lower()
                use_fsdp = env_fsdp in {"1", "true", "yes", "auto"}
                if (
                    use_fsdp
                    and FSDP is not None
                    and dist.is_available()
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                    and hasattr(model, "model")
                    and isinstance(model.model, torch.nn.Module)
                ):
                    try:
                        # Auto-disable if GPU memory is very small (<16GB)
                        if env_fsdp == "auto":
                            try:
                                if torch.cuda.is_available():
                                    props = torch.cuda.get_device_properties(0)
                                    if props.total_memory < 16 * 1024**3:
                                        use_fsdp = False
                                        logger.info("FSDP auto-disabled: GPU memory <16GB.")
                            except Exception:
                                pass
                        if not use_fsdp:
                            raise RuntimeError("FSDP auto disabled")

                        mp = None
                        if MixedPrecision is not None:
                            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                                mp = MixedPrecision(
                                    param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.bfloat16,
                                )
                            else:
                                mp = MixedPrecision(
                                    param_dtype=torch.float16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.float16,
                                )
                        shard = ShardingStrategy.FULL_SHARD if ShardingStrategy else None
                        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
                        model.model = FSDP(
                            model.model,
                            sharding_strategy=shard,
                            mixed_precision=mp,
                            device_id=local_rank if torch.cuda.is_available() else None,
                        )
                        logger.info(f"FSDP enabled for {name}")
                    except Exception as e:
                        logger.warning(f"FSDP wrap failed for {name}: {e}")

                # Boost Threads
                from ..core.system import resolve_cpu_budget, thread_limits

                try:
                    cpu_threads = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
                except Exception:
                    cpu_threads = 0
                if cpu_threads <= 0:
                    cpu_threads = resolve_cpu_budget()
                with thread_limits(blas_threads=cpu_threads):
                    # Fit logic (simplified here, assume factory configured it well)
                    fit_kwargs = {}
                    if hasattr(model, "fit"):
                        with contextlib.suppress(Exception):
                            sig = inspect.signature(model.fit)
                            if "tensorboard_writer" in sig.parameters:
                                fit_kwargs["tensorboard_writer"] = writer
                        if meta_fit_models is not None and self._fit_accepts_metadata(model):
                            fit_kwargs["metadata"] = meta_fit_models

                        model.fit(X_fit, y_fit, **fit_kwargs)

                # Only persist models that can actually predict (prevents "active but unusable" artifacts).
                sample_n = min(256, len(X_fit))
                sample_rows = np.arange(sample_n, dtype=np.int64)
                sample = self._slice_rows(X_fit, sample_rows)
                sample_meta = None
                if meta_fit_models is not None:
                    with contextlib.suppress(Exception):
                        sample_meta = self._slice_rows(meta_fit_models, sample_rows)
                ok, reason = self._smoke_predict(
                    model_name=name,
                    model=model,
                    sample_x=sample,
                    sample_meta=sample_meta,
                )
                if not ok:
                    logger.warning(f"{name} trained but is not usable for inference: {reason}")
                    self.run_summary.setdefault("model_health_failures", {})[name] = reason
                    continue

                if self._strict_model_check_enabled():
                    rt_ok, rt_reason = self._roundtrip_smoke_check(
                        model_name=name,
                        idx=idx,
                        model=model,
                        sample_x=sample,
                        sample_meta=sample_meta,
                    )
                    if not rt_ok:
                        logger.warning("%s failed save/load roundtrip check: %s", name, rt_reason)
                        self.run_summary.setdefault("model_health_failures", {})[name] = rt_reason
                        continue

                self.models[name] = model
                model.save(str(self.models_dir))

                # Prop Backtest Metric
                if meta_eval_models is not None and len(X_eval) > 0:
                    try:
                        pred_kwargs = {}
                        try:
                            import inspect

                            psig = inspect.signature(model.predict_proba)
                            has_kwargs = any(
                                p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in psig.parameters.values()  # noqa: B023
                            )
                            if meta_eval_models is not None and ("metadata" in psig.parameters or has_kwargs):
                                pred_kwargs["metadata"] = meta_eval_models
                        except Exception:
                            pass

                        probs_eval = self._pad_probs(model.predict_proba(X_eval, **pred_kwargs))
                        sig_eval = probs_to_signals(probs_eval).astype(np.int8, copy=False)
                        prop_metrics = prop_backtest(meta_eval_models, sig_eval)

                        fast_metrics: dict[str, Any] = {}
                        try:
                            close_vals = _frame_extract_column(meta_eval_models, "close")
                            high_vals = _frame_extract_column(meta_eval_models, "high")
                            low_vals = _frame_extract_column(meta_eval_models, "low")
                            if close_vals is not None and high_vals is not None and low_vals is not None:
                                close_arr = np.asarray(close_vals, dtype=np.float64).reshape(-1)
                                high_arr = np.asarray(high_vals, dtype=np.float64).reshape(-1)
                                low_arr = np.asarray(low_vals, dtype=np.float64).reshape(-1)
                                n_fast = int(min(close_arr.size, high_arr.size, low_arr.size, len(sig_eval)))
                                if n_fast > 0:
                                    close_arr = close_arr[:n_fast]
                                    high_arr = high_arr[:n_fast]
                                    low_arr = low_arr[:n_fast]
                                    sig_fast = np.asarray(sig_eval[:n_fast], dtype=np.int8)
                                    idx = _frame_index(meta_eval_models)
                                    if idx is None:
                                        idx = np.arange(n_fast, dtype=np.int64)
                                    month_idx, day_idx = self._month_day_indices_from_index(idx, n_fast)

                                    pip_size, pip_value_per_lot = infer_pip_metrics(self.settings.system.symbol)
                                    sl_cfg = getattr(self.settings.risk, "meta_label_sl_pips", None)
                                    tp_cfg = getattr(self.settings.risk, "meta_label_tp_pips", None)
                                    rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
                                    if sl_cfg is None or float(sl_cfg) <= 0:
                                        atr_raw = _frame_extract_column(meta_eval_models, "atr")
                                        atr_vals = None
                                        if atr_raw is not None:
                                            atr_vals = np.asarray(atr_raw, dtype=np.float64).reshape(-1)[:n_fast]
                                        open_raw = _frame_extract_column(meta_eval_models, "open")
                                        open_arr = (
                                            np.asarray(open_raw, dtype=np.float64).reshape(-1)[:n_fast]
                                            if open_raw is not None
                                            else close_arr
                                        )
                                        auto = infer_sl_tp_pips_auto(
                                            open_prices=open_arr,
                                            high_prices=high_arr,
                                            low_prices=low_arr,
                                            close_prices=close_arr,
                                            atr_values=atr_vals,
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
                                        close_prices=close_arr,
                                        high_prices=high_arr,
                                        low_prices=low_arr,
                                        signals=sig_fast,
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
                                    fast_metrics = {k: float(v) for k, v in zip(keys, arr.tolist(), strict=False)}
                        except Exception as e:
                            logger.debug(f"Fast backtest metrics failed for {name}: {e}")

                        self.run_summary.setdefault("model_metrics", {})[name] = {
                            "prop": prop_metrics,
                            "fast": fast_metrics,
                        }
                    except Exception as e:
                        logger.debug(f"Model backtest metrics failed for {name}: {e}")

            except Exception as e:
                logger.error(f"Training {name} failed: {e}")

            durations[name] = time.perf_counter() - t0
            # Cleanup
            import gc

            gc.collect()

        # 7. Meta Blender
        self._prune_low_quality_models()
        calibrators: dict[str, ProbabilityCalibrator] = {}
        if not stop_event or not stop_event.is_set():
            self._train_blender(X_eval, y_eval)
            calibrators = self._fit_probability_calibrators(X_eval, y_eval, meta_eval_models)
            self.run_summary["conformal_gate"] = self._fit_conformal_gate(X_eval, y_eval, meta_eval_models, calibrators)

        # 8. Evaluation (enforce walkforward/CPCV)
        if not stop_event or not stop_event.is_set():
            eval_index = _frame_index(X_eval)
            if eval_index is None:
                eval_index = np.arange(len(X_eval), dtype=np.int64)
            eval_dataset = PreparedDataset(
                X=X_eval,
                y=y_eval,
                index=eval_index,
                feature_names=list(_frame_columns(X_eval)),
                metadata=meta_eval_models,
                labels=y_eval,
            )
            self.run_summary["walkforward"] = self.evaluator.run_walkforward(
                eval_dataset, self.models, self._ensemble_proba, start_index=0
            )
            self.run_summary["cpcv"] = self.evaluator.run_cpcv(eval_dataset, self.models)

        # 9. Persistence
        self.run_summary["train_durations_sec"] = durations
        self.run_summary["train_samples"] = len(X_fit)
        self.run_summary["train_hardware"] = {
            "enable_gpu": bool(getattr(self.settings.system, "enable_gpu", False)),
            "num_gpus": int(getattr(self.settings.system, "num_gpus", 1)),
            "device": str(getattr(self.settings.system, "device", "cpu")),
        }
        if not self.distributed_enabled or self.rank == 0:
            self.persistence.save_run_summary(self.run_summary)
            self.persistence.save_active_models_list(self.models)
            self.persistence.save_models_bundle(self.models)
        if self.distributed_enabled:
            try:
                dist.barrier()
            except Exception:
                pass

        # 10. Export
        sample = self._slice_rows(X, np.arange(1, dtype=np.int64)) if len(X) > 0 else None
        if sample is not None and bool(getattr(self.settings.models, "export_onnx", False)):
            if not self.distributed_enabled or self.rank == 0:
                self.persistence.export_onnx(self.models, self.meta_blender, sample)
            if self.distributed_enabled:
                try:
                    dist.barrier()
                except Exception:
                    pass

        if writer:
            writer.close()
        self.persistence.cleanup_logs()
        logger.info("Training Complete.")

    def _prune_low_quality_models(self, dd_limit: float = 0.10, pfloor: float = 1.0) -> None:
        """
        Remove trained models that violate basic risk/quality thresholds before
        blending/export. This prevents obviously bad experts from dragging the
        ensemble or ONNX export.
        """
        try:
            metrics = self.run_summary.get("model_metrics", {})
        except Exception:
            metrics = {}

        removed: list[str] = []
        for name in list(self.models.keys()):
            m = metrics.get(name, {})
            fast = m.get("fast", m)
            pf = fast.get("profit_factor", None)
            dd = fast.get("max_dd", fast.get("max_dd_pct", None))
            try:
                if pf is not None and float(pf) <= pfloor:
                    removed.append(name)
                    continue
                if dd is not None and float(dd) > dd_limit:
                    removed.append(name)
            except Exception:
                continue

        for name in removed:
            self.models.pop(name, None)

        if removed:
            logger.info(f"Pruned low-quality models (pf<={pfloor} or dd>{dd_limit:.2%}): {removed}")

    def train_incremental(
        self, dataset: PreparedDataset, symbol: str, optimize: bool = False, stop_event: asyncio.Event | None = None
    ) -> None:
        """
        Incremental training wrapper.
        Reuses the factory logic but ensures models are loaded first.
        """
        if not self.models:
            self.load_models()

        X = dataset.X
        y = dataset.y if _is_series(dataset.y) else np.asarray(dataset.y, dtype=np.int8)
        if _frame_columns(X):
            selected = self._load_selected_features()
            if selected:
                X = self._align_feature_frame(X, selected)

        # Quick time estimate up front (works for CPU/GPU). Uses the same heuristic as full training.
        try:
            device = (
                "cuda"
                if bool(getattr(self.settings.system, "enable_gpu", False))
                else str(getattr(self.settings.system, "device", "cpu"))
            )
            bench_result = self.benchmarker.run_micro_benchmark(X, y, device)
            probe_x = self._slice_rows(X, np.arange(min(10_000, len(X)), dtype=np.int64))
            est_time = self.benchmarker.estimate_time(
                self._get_enabled_models(),
                len(X),
                bench_result,
                bool(getattr(self.settings.system, "enable_gpu", False)),
                int(getattr(self.settings.system, "num_gpus", 1)),
                context="incremental",
                historical_durations=self._historical_durations,
                historical_n=self._historical_samples,
                historical_gpu=(
                    self._historical_hardware.get("enable_gpu"),
                    self._historical_hardware.get("num_gpus", 1),
                ),
                incremental_stats=self.incremental_stats,
                probe_kwargs={
                    "X": probe_x,
                    "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                    "device": device,
                    "steps": 6,
                },
            )
            logger.info(
                f"Estimated incremental time for {symbol} (ALL {len(self._get_enabled_models())} models): "
                f"{est_time / 3600:.2f} hours ({est_time / 60:.0f} min) for {len(X):,} samples"
            )
            if len(self._get_enabled_models()) > 0:
                logger.info(f"  → Average per model: {est_time / len(self._get_enabled_models()) / 3600:.2f} hours")
            if est_time > 72 * 3600:
                logger.warning(
                    f"Estimate is very high ({est_time / 3600:.1f}h = {est_time / 3600 / 24:.1f} days). "
                    "Check GPU settings (enable_gpu/num_gpus/device)."
                )
        except Exception as e:
            logger.debug(f"Incremental time estimate failed: {e}")

        # Cast
        try:
            X = self._coerce_feature_container_float32(X)
        except Exception as e:
            logger.debug(f"Float32 casting failed in incremental training: {e}")

        enabled = self._get_enabled_models()
        enabled = self._maybe_shard_models(enabled)

        for idx, name in enumerate(enabled, 1):
            if stop_event and stop_event.is_set():
                break
            try:
                logger.info(f"Incremental: {name} on {symbol}")

                # Ensure model exists
                if name not in self.models:
                    self.models[name] = self.factory.create_model(name, {}, idx)

                model = self.models[name]

                from ..core.system import resolve_cpu_budget, thread_limits

                try:
                    cpu_threads = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
                except Exception:
                    cpu_threads = 0
                if cpu_threads <= 0:
                    cpu_threads = resolve_cpu_budget()
                with thread_limits(blas_threads=cpu_threads):
                    fit_kwargs = {}
                    # Inspect signature...
                    if hasattr(model, "fit"):
                        if self._fit_accepts_metadata(model):
                            fit_kwargs["metadata"] = dataset.metadata
                        t0 = time.perf_counter()
                        model.fit(X, y, **fit_kwargs)
                        duration = time.perf_counter() - t0
                        # Persist incremental timing for future estimates
                        self.incremental_stats[name] = {
                            "duration_sec": duration,
                            "n_samples": len(X),
                            "enable_gpu": bool(getattr(self.settings.system, "enable_gpu", False)),
                            "num_gpus": int(getattr(self.settings.system, "num_gpus", 1)),
                            "device": str(getattr(self.settings.system, "device", "cpu")),
                        }
                        self.persistence.save_incremental_stats(self.incremental_stats)

                sample_rows = np.arange(min(256, len(X)), dtype=np.int64)
                sample = self._slice_rows(X, sample_rows)
                sample_meta = None
                if dataset.metadata is not None:
                    with contextlib.suppress(Exception):
                        sample_meta = self._slice_rows(dataset.metadata, sample_rows)
                ok, reason = self._smoke_predict(
                    model_name=name,
                    model=model,
                    sample_x=sample,
                    sample_meta=sample_meta,
                )
                if not ok:
                    logger.warning("Inc train %s failed inference smoke check: %s", name, reason)
                    self.run_summary.setdefault("model_health_failures", {})[name] = reason
                    continue
                if self._strict_model_check_enabled():
                    rt_ok, rt_reason = self._roundtrip_smoke_check(
                        model_name=name,
                        idx=idx,
                        model=model,
                        sample_x=sample,
                        sample_meta=sample_meta,
                    )
                    if not rt_ok:
                        logger.warning("Inc train %s failed roundtrip check: %s", name, rt_reason)
                        self.run_summary.setdefault("model_health_failures", {})[name] = rt_reason
                        continue

                model.save(str(self.models_dir))
            except Exception as e:
                logger.warning(f"Inc train {name} failed: {e}")

        self.persistence.save_active_models_list(self.models)

    def load_models(self):
        self.models, self.meta_blender = self.persistence.load_models()

    def _get_enabled_models(self) -> list[str]:
        """
        Resolve enabled model names with conservative defaults for prop-style robustness:
        - honor explicit `ml_models`
        - always keep linear anchors
        - include online learners as add-ons (not replacements)
        - avoid implicitly forcing every experimental model unless explicitly requested
        """
        candidates = list(self.settings.models.ml_models)
        force_all = bool(getattr(self.settings.models, "train_all_registered_models", False))
        if force_all:
            candidates.extend(
                [
                    "transformer",
                    "patchtst",
                    "timesnet",
                    "evolution",
                    "genetic",
                    "rl_ppo",
                    "rl_sac",
                    "rllib_ppo",
                    "rllib_sac",
                    "nbeats",
                    "nbeatsx_nf",
                    "tide",
                    "tide_nf",
                    "tabnet",
                    "kan",
                    "elasticnet",
                    "bayes_logit",
                    "online_pa",
                    "online_hoeffding",
                    "vw",
                ]
            )
        else:
            if bool(getattr(self.settings.models, "ensure_linear_anchors", True)):
                candidates.extend(["elasticnet", "bayes_logit"])
            if bool(getattr(self.settings.models, "online_learners_enabled", True)):
                candidates.extend(["online_pa", "online_hoeffding"])

            # Include optional families only when their coarse feature flags are enabled.
            if bool(getattr(self.settings.models, "use_neuroevolution", False)):
                candidates.extend(["evolution", "genetic"])
            if bool(getattr(self.settings.models, "use_rl_agent", False)):
                candidates.append("rl_ppo")
            if bool(getattr(self.settings.models, "use_sac_agent", False)):
                candidates.append("rl_sac")
            if bool(getattr(self.settings.models, "use_rllib_agent", False)):
                candidates.extend(["rllib_ppo", "rllib_sac"])

        # Preserve order while deduplicating
        seen = set()
        ordered = []
        for m in candidates:
            norm = ModelFactory._normalize_model_key(m)
            if norm not in seen:
                ordered.append(norm)
                seen.add(norm)

        # Resolve strict-runtime redirects early so scheduling/filtering reflects the actual model class.
        with contextlib.suppress(Exception):
            from ..models.registry import _resolve_runtime_model_name  # type: ignore

            redirected: list[str] = []
            seen_redirect: set[str] = set()
            for name in ordered:
                resolved = _resolve_runtime_model_name(name)
                if resolved != name:
                    logger.info("Runtime model redirect in trainer: %s -> %s", name, resolved)
                if resolved not in seen_redirect:
                    redirected.append(resolved)
                    seen_redirect.add(resolved)
            ordered = redirected

        # Optional hard override for fast/diagnostic runs.
        override_raw = os.environ.get("FOREX_BOT_MODELS_OVERRIDE", "")
        if str(override_raw).strip():
            override: list[str] = []
            seen_override: set[str] = set()
            for token in str(override_raw).split(","):
                norm = ModelFactory._normalize_model_key(token.strip())
                with contextlib.suppress(Exception):
                    from ..models.registry import _resolve_runtime_model_name  # type: ignore

                    norm = _resolve_runtime_model_name(norm)
                if norm and norm not in seen_override:
                    override.append(norm)
                    seen_override.add(norm)
            if override:
                override_set = set(override)
                ordered = [m for m in ordered if m in override_set]
                logger.info("Model override active (FOREX_BOT_MODELS_OVERRIDE): %s", ordered)

        # Filter out models that cannot run in the current environment.
        # This keeps the configured/robust model set while avoiding known no-op experts
        # (e.g., RLlib on Python 3.13 Windows where Ray wheels do not exist).
        filtered: list[str] = []
        skipped: list[str] = []
        pandas_free = self._pandas_free_enabled()

        needed = set(ordered)

        RAY_AVAILABLE = False  # type: ignore
        if needed.intersection({"rllib_ppo", "rllib_sac"}):
            try:
                from ..models.rllib_agent import RAY_AVAILABLE  # type: ignore
            except Exception:
                RAY_AVAILABLE = False  # type: ignore

        SB3_AVAILABLE = False  # type: ignore
        if needed.intersection({"rl_ppo", "rl_sac"}):
            try:
                from ..models.rl import SB3_AVAILABLE  # type: ignore
            except Exception:
                SB3_AVAILABLE = False  # type: ignore

        FORECAST_NF_AVAILABLE = False  # type: ignore
        TRANSFORMER_NF_AVAILABLE = False  # type: ignore
        if not pandas_free:
            if needed.intersection({"tide_nf", "nbeatsx_nf"}):
                try:
                    from ..models.forecast_nf import NF_AVAILABLE as FORECAST_NF_AVAILABLE  # type: ignore
                except Exception:
                    FORECAST_NF_AVAILABLE = False  # type: ignore
            if needed.intersection({"patchtst", "timesnet"}):
                try:
                    from ..models.transformer_nf import NF_AVAILABLE as TRANSFORMER_NF_AVAILABLE  # type: ignore
                except Exception:
                    TRANSFORMER_NF_AVAILABLE = False  # type: ignore

        LINEAR_SKLEARN_AVAILABLE = False  # type: ignore
        LINEAR_VW_AVAILABLE = False  # type: ignore
        if needed.intersection({"elasticnet", "bayes_logit", "online_pa", "online_hoeffding", "vw"}):
            try:
                from ..models.linear import SKLEARN_AVAILABLE as LINEAR_SKLEARN_AVAILABLE  # type: ignore
                from ..models.linear import VW_AVAILABLE as LINEAR_VW_AVAILABLE  # type: ignore
            except Exception:
                LINEAR_SKLEARN_AVAILABLE = False  # type: ignore
                LINEAR_VW_AVAILABLE = False  # type: ignore

        try:
            from ..models.registry import MODEL_REGISTRY

            registry = set(MODEL_REGISTRY.keys())
        except Exception:
            registry = set()

        for name in ordered:
            if name in {"rllib_ppo", "rllib_sac"} and not bool(RAY_AVAILABLE):
                skipped.append(name)
                continue
            if name in {"rl_ppo", "rl_sac"} and not bool(SB3_AVAILABLE):
                skipped.append(name)
                continue
            if name in {"tide_nf", "nbeatsx_nf"} and not bool(FORECAST_NF_AVAILABLE):
                skipped.append(name)
                continue
            if name in {"patchtst", "timesnet"} and not bool(TRANSFORMER_NF_AVAILABLE):
                skipped.append(name)
                continue
            if name in {"elasticnet", "bayes_logit", "online_pa", "online_hoeffding"} and not bool(
                LINEAR_SKLEARN_AVAILABLE
            ):
                skipped.append(name)
                continue
            if name in {"vw"} and not bool(LINEAR_VW_AVAILABLE):
                skipped.append(name)
                continue
            if registry and name not in registry:
                skipped.append(name)
                continue
            filtered.append(name)

        if skipped:
            logger.info(f"Skipping unavailable models: {skipped}")

        if pandas_free:
            os.environ.setdefault("FOREX_BOT_TREE_BACKEND", "rust_strict")
            keep: list[str] = []
            dropped: list[str] = []
            dropped_missing: list[str] = []
            for m in list(filtered):
                if m not in _PANDAS_FREE_MODEL_ALLOWLIST:
                    dropped.append(m)
                    continue
                if m in _PANDAS_FREE_RUST_TREE_REQUIRED and not _rust_tree_binding_available(m):
                    dropped_missing.append(m)
                    continue
                keep.append(m)
            if dropped:
                logger.info("Pandas-free mode: skipping non-rust-tree models: %s", dropped)
            if dropped_missing:
                logger.info("Pandas-free mode: skipping tree models without Rust bindings: %s", dropped_missing)
            if not keep and filtered:
                logger.warning(
                    "Pandas-free mode: no Rust-backed models remain after filtering; "
                    "set FOREX_BOT_PANDAS_FREE=0 only if you intentionally want legacy python-frame fallback."
                )
            filtered = keep

        return filtered

    def _maybe_shard_models(self, models: list[str]) -> list[str]:
        """
        Hybrid Parallelism Strategy:
        - Deep Learning (GPU-heavy): Run on ALL ranks (Data Parallel).
        - Machine Learning (CPU-heavy): Shard across ranks (Task Parallel).
        """
        try:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                if world_size > 1:
                    deep_models = {
                        "transformer",
                        "patchtst",
                        "timesnet",
                        "kan",
                        "nbeats",
                        "nbeatsx_nf",
                        "tide",
                        "tide_nf",
                        "tabnet",
                        "rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac",
                        "genetic", "evolution" # GA runs independent islands, so all ranks ok
                    }

                    assigned = []
                    # Pre-compute sorted index map for O(1) lookup instead of O(n) per model
                    sorted_models = sorted(models)
                    index_map = {m: i for i, m in enumerate(sorted_models)}
                    for m in models:
                        if m in deep_models:
                            # Data Parallel: All ranks train this model together
                            # (Assuming model code handles DistributedSampler)
                            assigned.append(m)
                        else:
                            # Task Parallel: Split ML models across ranks
                            # deterministic assignment
                            idx = index_map[m]
                            if idx % world_size == rank:
                                assigned.append(m)

                    logger.info(f"Rank {rank}/{world_size} hybrid assignment: {assigned}")
                    return assigned
        except Exception as e:
            logger.warning(f"Model sharding check failed: {e}")
        return models

    def _train_blender(self, X, y):
        try:
            self.meta_blender = MetaBlender()
            X_m = X
            y_m = y
            if len(X_m) < 200 or len(y_m) < 200:
                return
            n_rows = int(len(X_m))
            y_arr = np.asarray(y_m, dtype=int).reshape(-1)
            if y_arr.size != n_rows:
                logger.warning("MetaBlender: X/y length mismatch (%s vs %s); skipping fit.", n_rows, y_arr.size)
                return
            model_buy: dict[str, np.ndarray] = {}
            model_acc: dict[str, float] = {}
            use_filter = bool(getattr(self.settings.models, "phase5_filter_meta_blender", False))
            core = set(getattr(self.settings.models, "phase5_core_models", []) or [])
            selected = self.models
            if use_filter and core:
                filtered = {name: m for name, m in self.models.items() if name in core}
                if len(filtered) >= 2:
                    selected = filtered
                    logger.info("MetaBlender: using Phase5 core models %s", sorted(filtered.keys()))
            contributed = 0
            for name, m in selected.items():
                try:
                    p = self._pad_probs(m.predict_proba(X_m))
                    if p.shape[0] != len(X_m):
                        continue
                    model_buy[name] = np.asarray(p[:, 1], dtype=np.float32)
                    pred_cls = np.argmax(p, axis=1)
                    pred_lbl = np.where(pred_cls == 2, -1, pred_cls).astype(int)
                    model_acc[name] = float((pred_lbl == y_arr).mean()) if len(y_arr) == len(pred_lbl) else 0.0
                    contributed += 1
                    # ...
                except Exception as exc:
                    logger.warning("MetaBlender: skipping model '%s': %s", name, exc)
            if contributed < 2:
                logger.warning("MetaBlender: insufficient valid base models (%s); skipping fit.", contributed)
                return

            drop: set[str] = set()
            if bool(getattr(self.settings.models, "phase5_diversity_filter", True)) and len(model_buy) >= 2:
                corr_thr = float(getattr(self.settings.models, "phase5_model_corr_max", 0.85) or 0.85)
                names = list(model_buy.keys())
                for i in range(len(names)):
                    a = names[i]
                    if a in drop:
                        continue
                    for j in range(i + 1, len(names)):
                        b = names[j]
                        if b in drop:
                            continue
                        try:
                            corr = float(np.corrcoef(model_buy[a], model_buy[b])[0, 1])
                        except Exception:
                            corr = 0.0
                        if not np.isfinite(corr):
                            corr = 0.0
                        if abs(corr) > corr_thr:
                            a_acc = float(model_acc.get(a, 0.0))
                            b_acc = float(model_acc.get(b, 0.0))
                            weaker = a if a_acc < b_acc else b
                            drop.add(weaker)
                if drop:
                    logger.info(
                        "MetaBlender diversity filter dropped %s highly correlated models: %s",
                        len(drop),
                        sorted(drop),
                    )

            kept_models = [name for name in model_buy.keys() if name not in drop]
            if len(kept_models) < 2:
                logger.warning("MetaBlender: <2 diverse models remain after correlation filter; skipping fit.")
                return

            X_meta = np.column_stack([model_buy[name] for name in kept_models]).astype(np.float32, copy=False)
            meta_features = [f"{name}_buy" for name in kept_models]
            payload: dict[str, Any] = {"X": X_meta, "y": y_arr, "feature_names": meta_features}
            index_obj = getattr(X_m, "index", None)
            if index_obj is not None:
                with contextlib.suppress(Exception):
                    idx_arr = np.asarray(index_obj).reshape(-1)
                    if idx_arr.size == X_meta.shape[0]:
                        payload["index"] = idx_arr

            self.meta_blender.fit(payload)
            self.meta_blender.save(self.models_dir / "meta_blender.joblib")
        except Exception as e:
            logger.error(f"Meta-blender training failed: {e}", exc_info=True)

    def _ensemble_proba(self, X):
        probs = []
        for m in self.models.values():
            try:
                probs.append(self._pad_probs(m.predict_proba(X)))
            except Exception as e:
                logger.debug(f"Model prediction failed in ensemble: {e}")
        if not probs:
            return np.zeros((len(X), 3))
        return np.mean(probs, axis=0)

    @staticmethod
    def _pad_probs(p):
        """
        HPC UNIFIED PROTOCOL: Force output to [Neutral, Buy, Sell].
        Standard indices: 0=Neutral, 1=Buy, 2=Sell.
        """
        if p is None:
            return np.zeros((0, 3))
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        n = p.shape[0]
        
        out = np.zeros((n, 3))
        if p.shape[1] == 3:
            return p # Assume standard protocol
        elif p.shape[1] == 2:
            # Model only knows Neutral vs Buy or similar
            out[:, 0] = p[:, 0]
            out[:, 1] = p[:, 1]
        else:
            out[:, 0] = 1.0 - p[:, 0]
            out[:, 1] = p[:, 0]
        return out

