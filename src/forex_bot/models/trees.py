from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import ExpertModel, get_early_stop_params, time_series_train_val_split, validate_time_ordering
from .label_utils import probs_to_three_class, remap_labels_sell_neutral_buy

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

try:
    import catboost as cb

    CAT_AVAILABLE = True
except Exception:
    cb = None
    CAT_AVAILABLE = False

import joblib

logger = logging.getLogger(__name__)


def _is_dataframe_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index"))


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))


def _is_datetime_index(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return True
    try:
        arr = np.asarray(value).reshape(-1)
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
    except Exception:
        return False


def _index_to_ns_int64(index: Any) -> np.ndarray | None:
    if index is None:
        return None
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
    if order.shape[0] != idx_ns.shape[0]:
        return None
    return order


def _sorted_time_order(index: Any) -> np.ndarray | None:
    idx_ns = _index_to_ns_int64(index)
    if idx_ns is None or idx_ns.size <= 1:
        return None
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None:
        return order
    return np.argsort(np.asarray(idx_ns, dtype=np.int64), kind="mergesort")


def _to_numpy_1d(values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        arr = np.asarray(values.to_numpy(copy=False), dtype=dtype)
    else:
        arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return np.asarray([arr.item()], dtype=dtype)
    return arr.reshape(-1)


def _pct_change(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.shape[0])
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    prev = arr[:-1]
    curr = arr[1:]
    delta = curr - prev
    np.divide(delta, prev, out=out[1:], where=np.abs(prev) > 1e-12)
    out[~np.isfinite(out)] = 0.0
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.shape[0])
    out = np.zeros(n, dtype=np.float64)
    w = max(1, int(window))
    if n <= 0 or n < w:
        return out
    arr2 = arr * arr
    c1 = np.cumsum(arr)
    c2 = np.cumsum(arr2)
    sum_w = c1[w - 1 :] - np.concatenate(([0.0], c1[:-w]))
    sq_w = c2[w - 1 :] - np.concatenate(([0.0], c2[:-w]))
    mean_w = sum_w / float(w)
    var_w = np.maximum((sq_w / float(w)) - (mean_w * mean_w), 0.0)
    out[w - 1 :] = np.sqrt(var_w)
    out[~np.isfinite(out)] = 0.0
    return out


def _feature_names(x: Any) -> list[str]:
    if hasattr(x, "columns"):
        return [str(c) for c in list(x.columns)]
    arr = np.asarray(x)
    if arr.ndim == 0:
        return ["f0"]
    if arr.ndim == 1:
        return ["f0"]
    return [f"f{i}" for i in range(int(arr.shape[1]))]


def _replace_inf_with_nan(x: Any) -> Any:
    if hasattr(x, "replace"):
        return x.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.where(np.isinf(arr), np.nan, arr)


def _sort_by_datetime_index(x: Any, y: Any) -> tuple[Any, Any]:
    idx = getattr(x, "index", None)
    if not _is_datetime_index(idx):
        return x, y
    try:
        monotonic_attr = getattr(idx, "is_monotonic_increasing", None)
        if monotonic_attr is None:
            idx_ns = _index_to_ns_int64(idx)
            is_monotonic = bool(
                idx_ns is not None and (idx_ns.size <= 1 or bool(np.all(idx_ns[1:] >= idx_ns[:-1])))
            )
        else:
            is_monotonic = bool(monotonic_attr)
        if not is_monotonic:
            order = _sorted_time_order(idx)
            if order is None:
                return x, y
            x = _slice_by_indices(x, order)
            y = _slice_by_indices(y, order)
    except Exception:
        return x, y
    return x, y


def _slice_by_indices(obj: Any, indices: Any) -> Any:
    if obj is None:
        return None
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if _is_dataframe_like(obj):
        with contextlib.suppress(Exception):
            return obj.take(idx)
        with contextlib.suppress(Exception):
            base_idx = np.asarray(getattr(obj, "index")).reshape(-1)
            if base_idx.shape[0] > 0:
                return obj.loc[base_idx[idx]]
    if _is_frame_like(obj):
        cols = getattr(obj, "columns", None)
        names: list[str] = []
        if cols is not None:
            with contextlib.suppress(Exception):
                names = [str(c) for c in list(cols)]
        out: dict[str, Any] = {}
        for col in names:
            with contextlib.suppress(Exception):
                vec = np.asarray(obj[col]).reshape(-1)  # type: ignore[index]
                out[col] = vec[idx]
        src_idx = getattr(obj, "index", None)
        if src_idx is not None:
            src_arr = np.asarray(src_idx).reshape(-1)
            out["index"] = src_arr[idx]
        return out
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    return arr[idx]


def _slice_rows(obj: Any, start: int, end: int | None = None) -> Any:
    if obj is None:
        return None
    s = max(0, int(start))
    if _is_dataframe_like(obj) or _is_frame_like(obj):
        n = int(len(obj))
        e = n if end is None else min(n, max(s, int(end)))
        return _slice_by_indices(obj, np.arange(s, e, dtype=np.int64))
    arr = np.asarray(obj)
    return arr[s:end] if end is not None else arr[s:]


def _cpu_threads_hint() -> int:
    try:
        return max(0, int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0))
    except Exception:
        return 0


def _tree_device_preference() -> str:
    """
    Controls whether tree models should run on GPU or CPU.

    Env:
      - FOREX_BOT_TREE_DEVICE=auto|gpu|cpu  (default: auto)
    """
    raw = str(os.environ.get("FOREX_BOT_TREE_DEVICE", "auto")).strip().lower()
    if raw in {"cpu", "gpu", "auto"}:
        return raw
    if raw in {"0", "false", "no", "off"}:
        return "cpu"
    if raw in {"1", "true", "yes", "on"}:
        return "gpu"
    return "auto"

def _gpu_only_mode() -> bool:
    raw = str(os.environ.get("FOREX_BOT_GPU_ONLY", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _quick_e2e_mode() -> bool:
    raw = str(os.environ.get("FOREX_BOT_QUICK_E2E", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _apply_quick_tree_caps(params: dict[str, Any], *, family: str) -> None:
    """
    Reduce heavy tree budgets during quick end-to-end validation runs.
    """
    if not _quick_e2e_mode():
        return

    try:
        if family in {"lgbm", "xgb"}:
            cap = int(os.environ.get("FOREX_BOT_QUICK_TREE_ESTIMATORS", "120") or 120)
            current = int(params.get("n_estimators", cap) or cap)
            params["n_estimators"] = max(16, min(current, cap))
            if "max_depth" in params:
                params["max_depth"] = max(2, min(int(params.get("max_depth", 6) or 6), 6))
            if "num_parallel_tree" in params:
                params["num_parallel_tree"] = max(1, min(int(params.get("num_parallel_tree", 2) or 2), 2))
            if str(params.get("booster", "")).strip().lower() == "dart":
                dart_cap = int(os.environ.get("FOREX_BOT_QUICK_XGB_DART_ESTIMATORS", "48") or 48)
                params["n_estimators"] = max(16, min(int(params.get("n_estimators", dart_cap) or dart_cap), dart_cap))

        if family == "cat":
            cap = int(os.environ.get("FOREX_BOT_QUICK_CAT_ITERATIONS", "120") or 120)
            current = int(params.get("iterations", cap) or cap)
            params["iterations"] = max(16, min(current, cap))
            if "depth" in params:
                params["depth"] = max(2, min(int(params.get("depth", 6) or 6), 6))
    except Exception:
        # Never fail model init due to quick-cap parsing.
        return


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _get_model_classes(model: Any) -> list[int] | None:
    """
    Best-effort extraction of class labels in the same order as predict_proba columns.
    Returns labels as ints when possible.
    """
    if model is None:
        return None
    with_classes = getattr(model, "classes_", None)
    if with_classes is not None:
        try:
            return [int(c) for c in list(with_classes)]
        except Exception:
            return None
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for key in ("classifier", "model"):
            step = named_steps.get(key)
            if step is not None and getattr(step, "classes_", None) is not None:
                try:
                    return [int(c) for c in list(step.classes_)]
                except Exception:
                    return None
    steps = getattr(model, "steps", None)
    if steps:
        try:
            _, last = steps[-1]
            if getattr(last, "classes_", None) is not None:
                return [int(c) for c in list(last.classes_)]
        except Exception:
            return None
    return None


def _reorder_to_neutral_buy_sell(probs: np.ndarray, classes: list[int] | None) -> np.ndarray:
    """
    HPC PROTOCOL: Force output to [Neutral, Buy, Sell].
    Standard indices: 0=Neutral, 1=Buy, 2=Sell.
    """
    return probs_to_three_class(
        probs,
        classes,
        class_to_output={0: 2, 1: 0, 2: 1},
    )


def _augment_time_features(df: Any) -> Any:
    """
    Add lightweight lag/volatility features for tree models when raw close is available.
    If 'close' is absent, returns the input unchanged.
    """
    if not _is_dataframe_like(df) or "close" not in df.columns:
        return df
    required = ("ret1", "ret1_lag1", "ret1_lag2", "ret1_lag5", "ret1_lag8", "vol14", "vol50", "mom5", "mom15")
    if all(col in df.columns for col in required):
        return df
    try:
        cached = df.attrs.get("_tree_augmented_cache")
        if _is_dataframe_like(cached):
            if all(col in cached.columns for col in required):
                return cached
    except Exception:
        pass
    try:
        out = df.copy()
        close = _to_numpy_1d(out["close"], dtype=np.float64)
        n = int(close.shape[0])
        ret1 = _pct_change(close)
        out["ret1"] = ret1
        out["ret1_lag1"] = np.concatenate(([0.0], ret1[:-1])) if n > 0 else ret1
        out["ret1_lag2"] = np.concatenate((np.zeros(2, dtype=np.float64), ret1[:-2])) if n > 2 else np.zeros(
            n, dtype=np.float64
        )
        out["ret1_lag5"] = np.concatenate((np.zeros(5, dtype=np.float64), ret1[:-5])) if n > 5 else np.zeros(
            n, dtype=np.float64
        )
        out["ret1_lag8"] = np.concatenate((np.zeros(8, dtype=np.float64), ret1[:-8])) if n > 8 else np.zeros(
            n, dtype=np.float64
        )
        out["vol14"] = _rolling_std(ret1, 14)
        out["vol50"] = _rolling_std(ret1, 50)
        mom5 = np.zeros(n, dtype=np.float64)
        mom15 = np.zeros(n, dtype=np.float64)
        if n > 5:
            mom5[5:] = close[5:] - close[:-5]
        if n > 15:
            mom15[15:] = close[15:] - close[:-15]
        out["mom5"] = mom5
        out["mom15"] = mom15
        try:
            df.attrs["_tree_augmented_cache"] = out
        except Exception:
            pass
        return out
    except Exception:
        return df


def _remap_labels_to_contiguous(y: Any | np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    HPC FIX: Hardcoded Deterministic Mapping.
    Prevents 'Label Drift' where Buy/Sell columns are swapped depending on data.
    Order: -1 -> 0 (Sell), 0 -> 1 (Neutral), 1 -> 2 (Buy)
    """
    mapping = {-1: 0, 0: 1, 1: 2}
    remapped = remap_labels_sell_neutral_buy(y)
    return remapped, mapping


def _resolve_lgbm_monotone_constraints(params: dict[str, Any], feature_names: list[str]) -> list[int] | None:
    def _to_constraint(v: Any) -> int:
        try:
            iv = int(v)
        except Exception:
            return 0
        if iv > 0:
            return 1
        if iv < 0:
            return -1
        return 0

    by_feature = params.pop("monotone_constraints_by_feature", None)
    if isinstance(by_feature, dict):
        vec = [_to_constraint(by_feature.get(str(col), 0)) for col in feature_names]
        return vec if any(v != 0 for v in vec) else None

    raw = params.get("monotone_constraints")
    if raw is None:
        return None

    vals: list[Any] = []
    if isinstance(raw, (list, tuple, np.ndarray)):
        vals = list(raw)
    elif isinstance(raw, str):
        txt = raw.strip()
        if txt:
            with_brackets = txt if txt.startswith("[") else f"[{txt}]"
            with contextlib.suppress(Exception):
                parsed = json.loads(with_brackets)
                if isinstance(parsed, list):
                    vals = parsed
            if not vals:
                vals = [p.strip() for p in txt.replace("[", "").replace("]", "").split(",")]

    if not vals:
        params.pop("monotone_constraints", None)
        return None

    if len(vals) < len(feature_names):
        vals = list(vals) + [0] * (len(feature_names) - len(vals))
    elif len(vals) > len(feature_names):
        vals = list(vals[: len(feature_names)])

    vec = [_to_constraint(v) for v in vals]
    return vec if any(v != 0 for v in vec) else None


class LightGBMExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx # Store index for GPU distribution
        self.params = params or {
            "n_estimators": 800,
            "num_leaves": 64,
            "learning_rate": 0.03,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            # Reduce overfitting, improve smoothness/extrapolation
            "min_data_in_leaf": 50,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "path_smooth": 10,
            "linear_tree": True,
        }
        _apply_quick_tree_caps(self.params, family="lgbm")

    def fit(self, x: Any, y: Any) -> bool:
        if not LGBM_AVAILABLE:
            logger.warning("LightGBM not available.")
            return False

        try:
            x = _augment_time_features(x)
            x, y = _sort_by_datetime_index(x, y)

            # Clean inf values (replace with nan)
            x = _replace_inf_with_nan(x)

            params = self.params.copy()
            mono = _resolve_lgbm_monotone_constraints(params, _feature_names(x))
            if mono is not None:
                params["monotone_constraints"] = mono
            else:
                params.pop("monotone_constraints", None)

            # Respect worker thread partitioning when set (prevents oversubscription on large CPUs).
            cpu_threads = _cpu_threads_hint()
            if cpu_threads > 0:
                params["n_jobs"] = cpu_threads

            pref = _tree_device_preference()
            has_cuda = _torch_cuda_available()
            requested = str(params.get("device_type", "")).strip().lower()

            if requested == "cpu" or pref == "cpu":
                use_gpu = False
            elif requested == "gpu" or pref == "gpu":
                use_gpu = has_cuda
                if not has_cuda:
                    logger.warning("LightGBM GPU requested but no CUDA devices detected; falling back to CPU.")
            else:
                # auto: prefer GPU if available
                use_gpu = has_cuda
            if _gpu_only_mode() and not use_gpu:
                logger.warning("GPU-only mode: LightGBM GPU unavailable; skipping LightGBM.")
                self._gpu_only_disabled = True
                self.model = None
                return False

            if use_gpu:
                params["device_type"] = "gpu"
                params.setdefault("max_bin", 63)
                params.setdefault("gpu_use_dp", False)
                # HPC: Spread models across all 8 GPUs
                import torch
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                params["gpu_device_id"] = (self.idx - 1) % gpu_count
            else:
                params["device_type"] = "cpu"
                params.pop("gpu_use_dp", None)
                params.pop("gpu_device_id", None)
                # max_bin=255 for better accuracy on CPU
                params.setdefault("max_bin", 255)

            # Use time-series aware split to prevent look-ahead bias
            if len(y) > 500:
                try:
                    validate_time_ordering(x, context="LightGBMExpert.fit")
                    y_arr, mapping = _remap_labels_to_contiguous(y)
                    embargo = max(24, int(len(y_arr) * 0.01))
                    x_train, x_val, y_train, y_val = time_series_train_val_split(
                        x, y_arr, val_ratio=0.15, min_train_samples=100, embargo_samples=embargo
                    )
                    eval_set = [(x_val, y_val)]
                except ValueError:
                    # Fall back to simple positional split if validation fails
                    split_idx = int(len(x) * 0.85)
                    y_arr, mapping = _remap_labels_to_contiguous(y)
                    x_train, x_val = _slice_rows(x, 0, split_idx), _slice_rows(x, split_idx, None)
                    y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
                    eval_set = [(x_val, y_val)]
            else:
                y_arr, mapping = _remap_labels_to_contiguous(y)
                x_train, y_train = x, y_arr
                eval_set = [(x, y_arr)]

            # Build class weights on remapped labels so keys align with LightGBM class map.
            uniq, counts = np.unique(y_train, return_counts=True)
            class_weight = {
                int(cls): float(len(y_train) / (len(uniq) * cnt))
                for cls, cnt in zip(uniq, counts, strict=False)
                if cnt > 0
            }
            params["class_weight"] = class_weight if class_weight else None

            # Decide binary vs multiclass based on remapped labels
            binary = len(uniq) <= 2
            if binary:
                params["objective"] = "binary"
                params.pop("num_class", None)
                eval_metric = "binary_logloss"
            else:
                params["objective"] = "multiclass"
                params["num_class"] = params.get("num_class", len(uniq))
                eval_metric = "multi_logloss"

            self.model = lgb.LGBMClassifier(**params)
            try:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=eval_set,
                    eval_metric=eval_metric,
                    callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
                )
            except Exception as e:
                if params.get("device_type") == "gpu":
                    if _gpu_only_mode():
                        logger.warning(f"GPU-only mode: LightGBM GPU training failed ({e}); skipping LightGBM.")
                        self._gpu_only_disabled = True
                        self.model = None
                        return False
                    logger.warning(f"LightGBM GPU training failed ({e}), falling back to CPU.", exc_info=True)
                    params["device_type"] = "cpu"
                    params.pop("gpu_use_dp", None)
                    params.pop("gpu_device_id", None)
                    params.pop("max_bin", None)
                    params.setdefault("max_bin", 255)
                    self.model = lgb.LGBMClassifier(**params)
                    self.model.fit(
                        x_train,
                        y_train,
                        eval_set=eval_set,
                        eval_metric=eval_metric,
                        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
                    )
                else:
                    raise e
            return True
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}", exc_info=True)
            self.model = None
            return False

    def predict_proba(self, x: Any) -> np.ndarray:
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: LightGBM skipped.")
        if self.model is None:
            raise RuntimeError("LightGBM model not loaded")
        try:
            x = _augment_time_features(x)
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception as exc:
            logger.error(f"LightGBM inference failed: {exc}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "lightgbm.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "lightgbm.joblib"
        if p.exists():
            self.model = joblib.load(p)


class XGBoostExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx

        self.params = params or {
            "n_estimators": 800,
            "max_depth": 8,
            "learning_rate": 0.05,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "tree_method": "hist",  # default; overridden to gpu_hist if GPU
        }
        _apply_quick_tree_caps(self.params, family="xgb")

        cpu_threads = _cpu_threads_hint()
        if cpu_threads > 0 and int(self.params.get("n_jobs", -1) or -1) < 0:
            self.params["n_jobs"] = cpu_threads

        pref = _tree_device_preference()
        has_cuda = _torch_cuda_available()
        requested = str(self.params.get("device", "")).strip().lower()

        if requested.startswith("cuda"):
            use_gpu = has_cuda
        elif pref == "cpu":
            use_gpu = False
        elif pref == "gpu":
            use_gpu = has_cuda
        else:
            # auto: prefer GPU if available
            use_gpu = has_cuda

        if use_gpu:
            # Newer XGBoost prefers `tree_method=hist` + `device=cuda`.
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_id = (self.idx - 1) % gpu_count
            self.params.setdefault("device", f"cuda:{gpu_id}")
            self.params["tree_method"] = "gpu_hist"
        else:
            self.params.pop("device", None)
            self.params["tree_method"] = "hist"
            if _gpu_only_mode():
                self._gpu_only_disabled = True

    def fit(self, x: Any, y: Any) -> None:
        if self._gpu_only_disabled:
            logger.warning("GPU-only mode: XGBoost GPU unavailable; skipping.")
            self.model = None
            return
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available")
            return
        try:
            x, y = _sort_by_datetime_index(x, y)

            # XGBoost can't handle inf values - replace with nan (which XGBoost handles natively)
            x = _replace_inf_with_nan(x)

            # XGBoost's sklearn wrapper expects multiclass labels to be contiguous: {0,1,2,...}.
            # Our project uses {-1,0,1} where -1=sell, 0=neutral, 1=buy.
            y_arr = np.asarray(y, dtype=int)
            # Backward compat: some older pipelines used 2=sell.
            y_arr = np.where(y_arr == -1, 2, y_arr).astype(int, copy=False)

            x_train = x
            y_train = y_arr
            eval_set = None
            if len(y_arr) > 500:
                try:
                    validate_time_ordering(x, context="XGBoostExpert.fit")
                    embargo = max(24, int(len(y_arr) * 0.01))
                    x_train, x_val, y_train_s, y_val_s = time_series_train_val_split(
                        x, y_arr, val_ratio=0.15, min_train_samples=100, embargo_samples=embargo
                    )
                    y_train = np.asarray(y_train_s, dtype=int)
                    y_val = np.asarray(y_val_s, dtype=int)
                    eval_set = [(x_val, y_val)]
                except ValueError:
                    split_idx = int(len(x) * 0.85)
                    x_train, x_val = _slice_rows(x, 0, split_idx), _slice_rows(x, split_idx, None)
                    y_train = y_arr[:split_idx]
                    y_val = y_arr[split_idx:]
                    eval_set = [(x_val, y_val)]

            uniq, counts = np.unique(y_train, return_counts=True)
            sample_weight = np.ones(len(y_train))
            for cls, cnt in zip(uniq, counts, strict=False):
                if cnt > 0:
                    sample_weight[y_train == cls] = len(y_train) / (len(uniq) * cnt)

            # XGBoost 2.0+: early_stopping_rounds moved from fit() to constructor
            params = self.params.copy()
            fit_kwargs = {}
            if eval_set:
                es_pat, _ = get_early_stop_params(50, 0.0)
                if es_pat > 0:
                    params["early_stopping_rounds"] = es_pat
                    fit_kwargs["eval_set"] = eval_set
                    fit_kwargs["verbose"] = False

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(x_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    def predict_proba(self, x: Any) -> np.ndarray:
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: XGBoost skipped.")
        if self.model is None:
            return np.zeros((len(x), 3))
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception:
            return np.zeros((len(x), 3))

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "xgboost.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "xgboost.joblib"
        if p.exists():
            self.model = joblib.load(p)


class CatBoostExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx

        pref = _tree_device_preference()
        has_cuda = _torch_cuda_available()

        self.params = params or {
            "iterations": 800,
            "depth": 8,
            "learning_rate": 0.05,
            "loss_function": "MultiClass",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        }
        _apply_quick_tree_caps(self.params, family="cat")

        cpu_threads = _cpu_threads_hint()
        if cpu_threads > 0 and int(self.params.get("thread_count", -1) or -1) < 0:
            self.params["thread_count"] = cpu_threads

        requested = str(self.params.get("task_type", "")).strip().lower()
        if requested == "gpu":
            use_gpu = has_cuda
        elif pref == "cpu":
            use_gpu = False
        elif pref == "gpu":
            use_gpu = has_cuda
        else:
            # auto: prefer GPU if available
            use_gpu = has_cuda

        if use_gpu:
            self.params.setdefault("task_type", "GPU")
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_id = (self.idx - 1) % gpu_count
            self.params.setdefault("devices", str(gpu_id))
            self.params.setdefault("border_count", 32)
        else:
            # Ensure we don't accidentally try to run CatBoost in GPU mode without devices.
            if str(self.params.get("task_type", "")).strip().lower() == "gpu":
                self.params.pop("task_type", None)
                self.params.pop("devices", None)
            if _gpu_only_mode():
                self._gpu_only_disabled = True

    def fit(self, x: Any, y: Any) -> None:
        if self._gpu_only_disabled:
            logger.warning("GPU-only mode: CatBoost GPU unavailable; skipping.")
            self.model = None
            return
        if not CAT_AVAILABLE:
            logger.warning("CatBoost not available")
            return
        try:
            x, y = _sort_by_datetime_index(x, y)

            # Clean inf values (replace with nan)
            x = _replace_inf_with_nan(x)

            y_arr = np.asarray(y, dtype=int).reshape(-1)
            uniq, counts = np.unique(y_arr, return_counts=True)
            class_weights = {
                int(cls): float(len(y_arr) / (len(uniq) * cnt))
                for cls, cnt in zip(uniq, counts, strict=False)
                if cnt > 0
            }

            params = self.params.copy()
            params["class_weights"] = list(class_weights.values()) if class_weights else None

            x_train = x
            y_train = y_arr
            eval_set = None
            if len(y_arr) > 500:
                try:
                    validate_time_ordering(x, context="CatBoostExpert.fit")
                    embargo = max(24, int(len(y_arr) * 0.01))
                    x_train, x_val, y_train_s, y_val_s = time_series_train_val_split(
                        x,
                        y_arr,
                        val_ratio=0.15,
                        min_train_samples=100,
                        embargo_samples=embargo,
                    )
                    y_train = np.asarray(y_train_s, dtype=int)
                    y_val = np.asarray(y_val_s, dtype=int)
                    eval_set = (x_val, y_val)
                except ValueError:
                    split_idx = int(len(x) * 0.85)
                    x_train, x_val = _slice_rows(x, 0, split_idx), _slice_rows(x, split_idx, None)
                    y_train = y_arr[:split_idx]
                    y_val = y_arr[split_idx:]
                    eval_set = (x_val, y_val)

            if eval_set is not None:
                es_pat, _ = get_early_stop_params(50, 0.0)
                if es_pat > 0:
                    params.setdefault("od_type", "Iter")
                    params.setdefault("od_wait", int(es_pat))
                    params.setdefault("use_best_model", True)

            self.model = cb.CatBoostClassifier(**params)
            if eval_set is not None:
                self.model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(x_train, y_train)
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")

    def predict_proba(self, x: Any) -> np.ndarray:
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: CatBoost skipped.")
        if self.model is None:
            return np.zeros((len(x), 3))
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception:
            return np.zeros((len(x), 3))

    def save(self, path: str) -> None:
        if self.model:
            # CatBoost has its own efficient format
            self.model.save_model(str(Path(path) / "catboost.cbm"))

    def load(self, path: str) -> None:
        if not CAT_AVAILABLE:
            return
        p = Path(path) / "catboost.cbm"
        if p.exists():
            try:
                self.model = cb.CatBoostClassifier()
                self.model.load_model(str(p))
            except Exception as e:
                logger.warning(f"Failed to load CatBoost: {e}")


class CatBoostAltExpert(CatBoostExpert):
    """Alternate CatBoost preset (CPU/GPU capable) to diversify tree ensemble."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "iterations": 900,
            "depth": 10,
            "learning_rate": 0.03,
            "loss_function": "MultiClass",
            "random_seed": 7,
            "verbose": False,
            "thread_count": -1,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.5,
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)


class XGBoostRFExpert(XGBoostExpert):
    """Random-forest flavored XGBoost (CPU/GPU via XGBoost)."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.3,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "colsample_bytree": 0.8,
            "num_parallel_tree": 8,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)


class XGBoostDARTExpert(XGBoostExpert):
    """DART (dropout) XGBoost variant (CPU/GPU via XGBoost)."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "n_estimators": 600,
            "max_depth": 8,
            "learning_rate": 0.05,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "booster": "dart",
            "rate_drop": 0.10,
            "skip_drop": 0.50,
            "sample_type": "uniform",
            "normalize_type": "tree",
            "tree_method": "hist",
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)


