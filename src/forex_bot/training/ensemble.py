from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..models.base import validate_time_ordering

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:
    GradientBoostingClassifier = None
    LogisticRegression = None

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


logger = logging.getLogger(__name__)


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


def _rust_sort_rows_with_labels_by_index(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    index: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "sort_rows_with_labels_by_index"):
        return None
    idx_ns = _index_to_ns_int64(index)
    if idx_ns is None:
        return None
    rows = int(min(x_arr.shape[0], y_arr.shape[0], idx_ns.shape[0]))
    if rows <= 0:
        return np.zeros((0, x_arr.shape[1]), dtype=np.float32), np.zeros(0, dtype=y_arr.dtype)
    try:
        out_x, out_y, _out_idx = _fb.sort_rows_with_labels_by_index(
            np.asarray(x_arr[:rows], dtype=np.float32, order="C"),
            np.asarray(y_arr[:rows], dtype=np.int64),
            np.asarray(idx_ns[:rows], dtype=np.int64),
        )
    except Exception:
        return None
    x_sorted = np.asarray(out_x, dtype=np.float32)
    y_sorted_i64 = np.asarray(out_y, dtype=np.int64).reshape(-1)
    if x_sorted.ndim != 2 or x_sorted.shape[0] != rows or y_sorted_i64.shape[0] != rows:
        return None
    return x_sorted, y_sorted_i64.astype(y_arr.dtype, copy=False)


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


def _sorted_time_order(index: Any, n_rows: int) -> np.ndarray | None:
    idx_ns = _index_to_ns_int64(index)
    if idx_ns is None or idx_ns.size < int(n_rows) or int(n_rows) <= 1:
        return None
    idx_ns = np.asarray(idx_ns[: int(n_rows)], dtype=np.int64)
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None and order.size == int(n_rows):
        return order
    return np.argsort(idx_ns, kind="mergesort")


def _is_dataframe_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "iloc"))


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "__getitem__"))


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


def pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    Normalize probability outputs to [neutral, buy, sell].
    """
    if probs is None or len(probs) == 0:
        return np.zeros((0, 3), dtype=float)

    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    out = np.zeros((n, 3), dtype=float)

    if classes is not None and len(classes) == arr.shape[1]:
        for col, cls_val in enumerate(classes):
            if cls_val == 0:
                out[:, 0] = arr[:, col]
            elif cls_val == 1:
                out[:, 1] = arr[:, col]
            elif cls_val in (-1, 2):
                out[:, 2] = arr[:, col]
        return out

    if arr.shape[1] == 3:
        return arr
    if arr.shape[1] == 2:
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out

    out[:, 0] = 1.0 - arr[:, 0]
    out[:, 1] = arr[:, 0]
    return out


class MetaBlender:
    """Meta-level blending for specialist models (Stacking)."""

    def __init__(self) -> None:
        if XGB_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="mlogloss", n_jobs=-1, random_state=42
            )
        elif GradientBoostingClassifier:
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        elif LogisticRegression:
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        else:
            self.model = None

        self.feature_columns: list[str] | None = None
        self.proba_classes: list[int] | None = None
        self.constant_proba: np.ndarray | None = None

    @staticmethod
    def _as_2d_float32(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not arr.flags.writeable:
            arr = arr.copy()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    @staticmethod
    def _default_feature_names(n_features: int) -> list[str]:
        return [f"f{i}" for i in range(int(max(0, n_features)))]

    @staticmethod
    def _as_1d(values: Any) -> np.ndarray:
        if hasattr(values, "to_numpy"):
            try:
                arr = values.to_numpy(copy=False)
            except Exception:
                arr = np.asarray(values)
        else:
            arr = np.asarray(values)
        return np.asarray(arr).reshape(-1)

    @staticmethod
    def _as_numeric_1d(values: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
        arr = MetaBlender._as_1d(values)
        if arr.size <= 0:
            return arr.astype(dtype, copy=False)
        if arr.dtype.kind in {"b", "i", "u", "f"}:
            out = arr.astype(dtype, copy=False)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.empty(arr.size, dtype=dtype)
        string_codes: dict[str, float] = {}
        for i, value in enumerate(arr.tolist()):
            try:
                out[i] = float(value)
            except Exception:
                key = str(value)
                if key not in string_codes:
                    string_codes[key] = float(len(string_codes))
                out[i] = string_codes[key]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _fit_length(arrays: list[np.ndarray], fallback: int = 0) -> int:
        if not arrays:
            return int(max(0, fallback))
        lengths = [int(np.asarray(a).reshape(-1).size) for a in arrays if np.asarray(a).size > 0]
        if not lengths:
            return int(max(0, fallback))
        return int(min(lengths))

    def _coerce_fit_inputs(self, frame: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
        raw_index = None
        if _is_dataframe_like(frame):
            dataset = frame.copy()
            if "label" not in dataset:
                raise ValueError("Dataset must include 'label' column")

            # Enforce chronological ordering for time-series splits (prevents look-ahead bias).
            if hasattr(dataset.index, "is_monotonic_increasing") and not dataset.index.is_monotonic_increasing:
                try:
                    dataset = dataset.sort_index(kind="mergesort")
                except Exception as exc:
                    raise ValueError(
                        "MetaBlender.fit: Data index is NOT monotonically increasing and could not be sorted. "
                        "Time-series models require chronologically ordered data to prevent look-ahead bias."
                    ) from exc

            validate_time_ordering(dataset, context="MetaBlender.fit")

            labels_raw = dataset.pop("label").astype(int).to_numpy()
            if "symbol" in dataset:
                symbols = dataset.pop("symbol")
                symbols_arr = np.asarray(symbols).reshape(-1)
                symbols_str = symbols_arr.astype(str, copy=False)
                for sym in sorted({str(v) for v in symbols_str.tolist()}):
                    dataset[f"sym_{sym}"] = (symbols_str == sym).astype(np.float32, copy=False)

            regime_cols = [c for c in dataset.columns if str(c).startswith("regime_p")]
            if regime_cols:
                for col in regime_cols:
                    dataset[col] = dataset[col].astype(np.float32)
            elif "regime" in dataset:
                dataset["regime"] = dataset["regime"].astype("category")
                if XGB_AVAILABLE and hasattr(self.model, "enable_categorical"):
                    self.model.set_params(enable_categorical=True)

            features = dataset.reset_index(drop=True).fillna(0.0)
            feature_cols = [str(c) for c in list(features.columns)]
            X_arr = self._as_2d_float32(features.to_numpy(dtype=np.float32, copy=False))
            y_arr = np.asarray(labels_raw, dtype=int).reshape(-1)
            return X_arr, y_arr, feature_cols
        if _is_frame_like(frame):
            cols = _frame_columns(frame)
            label_col = _frame_resolve_column(frame, "label")
            if label_col is None:
                raise ValueError("Dataset must include 'label' column")

            raw_index = getattr(frame, "index", None)
            labels_raw = np.asarray(self._as_1d(frame[label_col]), dtype=int).reshape(-1)
            symbol_col = _frame_resolve_column(frame, "symbol")
            symbols_str: np.ndarray | None = None
            if symbol_col is not None:
                symbols_str = self._as_1d(frame[symbol_col]).astype(str, copy=False)

            feature_cols: list[str] = []
            vectors: list[np.ndarray] = []
            for col in cols:
                if col == label_col:
                    continue
                if symbol_col is not None and col == symbol_col:
                    continue
                feature_cols.append(str(col))
                vectors.append(self._as_numeric_1d(frame[col], dtype=np.float32))

            if symbols_str is not None and symbols_str.size > 0:
                for sym in sorted({str(v) for v in symbols_str.tolist()}):
                    feature_cols.append(f"sym_{sym}")
                    vectors.append((symbols_str == sym).astype(np.float32, copy=False))

            n_rows = self._fit_length([labels_raw, *vectors], fallback=int(labels_raw.size))
            labels_raw = labels_raw[:n_rows]
            if vectors:
                X_arr = np.column_stack([np.asarray(vec, dtype=np.float32).reshape(-1)[:n_rows] for vec in vectors])
            else:
                X_arr = np.zeros((n_rows, 0), dtype=np.float32)
            y_arr = np.asarray(labels_raw, dtype=int).reshape(-1)

            if raw_index is not None:
                sorted_xy = _rust_sort_rows_with_labels_by_index(X_arr, y_arr, raw_index)
                if sorted_xy is not None:
                    X_arr, y_arr = sorted_xy
                else:
                    try:
                        order = _sorted_time_order(raw_index, n_rows)
                        if order is not None:
                            X_arr = X_arr[order]
                            y_arr = y_arr[order]
                    except Exception:
                        pass

            if len(feature_cols) != X_arr.shape[1]:
                feature_cols = self._default_feature_names(X_arr.shape[1])
            return self._as_2d_float32(X_arr), y_arr, feature_cols

        raw_X = None
        raw_y = None
        feature_cols: list[str] = []
        if isinstance(frame, dict):
            if "X" in frame:
                raw_X = frame.get("X")
                raw_y = frame.get("y", frame.get("label"))
                raw_index = frame.get("index")
                names = frame.get("feature_names")
                if names is not None:
                    feature_cols = [str(c) for c in list(names)]
            else:
                if "label" not in frame:
                    raise ValueError("Dataset must include 'label' column")
                raw_y = frame.get("label")
                raw_index = frame.get("index")
                feature_cols = [str(k) for k in frame.keys() if str(k) != "label"]
                if feature_cols:
                    cols = [self._as_numeric_1d(frame.get(col), dtype=np.float32).reshape(-1) for col in feature_cols]
                    lengths = {int(c.size) for c in cols}
                    if len(lengths) > 1:
                        raise ValueError("All feature columns must have identical length")
                    raw_X = np.column_stack(cols)
                else:
                    n = int(np.asarray(raw_y).reshape(-1).shape[0])
                    raw_X = np.zeros((n, 0), dtype=np.float32)
        elif isinstance(frame, (tuple, list)) and len(frame) == 2:
            raw_X, raw_y = frame[0], frame[1]
        else:
            raise ValueError("MetaBlender.fit expects frame-like data, dict, or (X, y) tuple")

        if raw_X is None or raw_y is None:
            raise ValueError("MetaBlender.fit missing X/y inputs")
        X_arr = self._as_2d_float32(raw_X)
        y_arr = np.asarray(raw_y, dtype=int).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("MetaBlender.fit: X/y length mismatch")
        if raw_index is not None:
            sorted_xy = _rust_sort_rows_with_labels_by_index(X_arr, y_arr, raw_index)
            if sorted_xy is not None:
                X_arr, y_arr = sorted_xy
            else:
                try:
                    order = _sorted_time_order(raw_index, X_arr.shape[0])
                    if order is not None:
                        X_arr = X_arr[order]
                        y_arr = y_arr[order]
                except Exception:
                    pass
        if len(feature_cols) != X_arr.shape[1]:
            feature_cols = self._default_feature_names(X_arr.shape[1])
        return X_arr, y_arr, feature_cols

    def fit(self, frame: Any, val_ratio: float = 0.15) -> dict[str, float]:
        """
        Fit meta-blender using time-series aware validation.

        IMPORTANT: The input frame should contain out-of-fold predictions from
        base models (obtained via cross-validation or time-series split).
        If base predictions were obtained on the same data used here,
        there will be data leakage.

        Parameters
        ----------
        frame : Any
            Frame-like data with base model predictions and 'label' column
        val_ratio : float
            Fraction of data for validation (time-series split, not random)

        Returns
        -------
        dict
            Training metrics including validation accuracy
        """
        if self.model is None:
            return {"error": "No sklearn available"}

        features, labels_raw, feature_cols = self._coerce_fit_inputs(frame)
        if len(features) < 2:
            raise ValueError("MetaBlender.fit requires at least 2 samples")

        # Backward compat: some older components used 2=sell; normalize to -1.
        labels_canon = np.where(labels_raw == 2, -1, labels_raw).astype(int, copy=False)
        unique = sorted({int(v) for v in labels_canon.tolist()})
        # Some estimators (notably XGBoost multi-class) require labels 0..K-1.
        to_model = {lab: i for i, lab in enumerate(unique)}
        y_model = np.array([to_model[int(v)] for v in labels_canon], dtype=int)
        self.constant_proba = None
        self.feature_columns = list(feature_cols)

        # Time-series train/val split (no random shuffle!)
        n = int(len(features))
        val_size = max(1, int(n * val_ratio))
        val_size = min(max(1, n - 1), val_size)
        train_end = n - val_size

        X_train = features[:train_end]
        y_train = y_model[:train_end]
        X_val = features[train_end:]
        y_val = y_model[train_end:]

        # Degenerate train split can happen on short/sparse runs; keep blender functional.
        if len(np.unique(y_train)) < 2:
            counts = {int(k): 0 for k in (-1, 0, 1)}
            vals, cnts = np.unique(labels_canon, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                counts[int(v)] = int(c)
            total = float(sum(counts.values()) + 3)  # Laplace smoothing
            p_neutral = (counts[0] + 1) / total
            p_buy = (counts[1] + 1) / total
            p_sell = (counts[-1] + 1) / total
            self.constant_proba = np.array([p_neutral, p_buy, p_sell], dtype=float)
            self.proba_classes = [0, 1, -1]
            logger.warning(
                "MetaBlender: single-class train split; using constant fallback probs "
                "(neutral=%.3f buy=%.3f sell=%.3f).",
                p_neutral,
                p_buy,
                p_sell,
            )
            return {
                "samples": int(len(y_model)),
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "train_accuracy": 0.0,
                "val_accuracy": 0.0,
                "accuracy": 0.0,
                "classes": list(self.proba_classes),
            }

        self.model.fit(X_train, y_train)
        try:
            classes_model = [int(c) for c in list(self.model.classes_)]
            self.proba_classes = [unique[c] for c in classes_model]
        except Exception:
            self.proba_classes = unique

        # Training accuracy (in-sample, for diagnostics only)
        train_preds = self.model.predict(X_train)
        train_accuracy = float(np.mean(train_preds == y_train)) if len(y_train) else 0.0

        # Validation accuracy (out-of-sample, the real metric)
        val_preds = self.model.predict(X_val)
        val_accuracy = float(np.mean(val_preds == y_val)) if len(y_val) else 0.0

        logger.info(
            f"MetaBlender: train_acc={train_accuracy:.3f}, val_acc={val_accuracy:.3f}, "
            f"train_n={len(y_train)}, val_n={len(y_val)}"
        )

        return {
            "samples": int(len(y_model)),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "accuracy": val_accuracy,  # Report val accuracy as primary metric
            "classes": list(self.proba_classes or []),
        }

    def _map_named_features(
        self,
        X_arr: np.ndarray,
        input_names: list[str],
        *,
        symbols: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("MetaBlender has not been fitted")
        out = np.zeros((int(X_arr.shape[0]), int(len(self.feature_columns))), dtype=np.float32)
        idx_map = {str(col): i for i, col in enumerate(input_names)}
        for i, col in enumerate(self.feature_columns):
            j = idx_map.get(str(col))
            if j is not None and j < X_arr.shape[1]:
                out[:, i] = X_arr[:, j]
                continue
            if symbols is not None and str(col).startswith("sym_"):
                out[:, i] = (symbols == str(col)[4:]).astype(np.float32, copy=False)
        return out

    def _coerce_predict_input(self, frame: Any) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("MetaBlender has not been fitted")

        input_names: list[str] = []
        symbols: np.ndarray | None = None
        raw_X: Any = frame

        if _is_dataframe_like(frame):
            input_names = [str(c) for c in list(frame.columns)]
            if "symbol" in frame.columns:
                try:
                    symbols = frame["symbol"].astype(str).to_numpy()
                except Exception:
                    symbols = np.asarray(frame["symbol"], dtype=str).reshape(-1)
            raw_X = frame.to_numpy(dtype=np.float32, copy=False)
        elif _is_frame_like(frame):
            symbol_col = _frame_resolve_column(frame, "symbol")
            cols = _frame_columns(frame)
            vectors: list[np.ndarray] = []
            lengths: list[int] = []
            for col in cols:
                if symbol_col is not None and col == symbol_col:
                    continue
                vec = self._as_numeric_1d(frame[col], dtype=np.float32).reshape(-1)
                vectors.append(vec)
                input_names.append(str(col))
                lengths.append(int(vec.size))
            n_rows = min(lengths) if lengths else int(len(frame))
            if vectors:
                raw_X = np.column_stack([vec[:n_rows] for vec in vectors])
            else:
                raw_X = np.zeros((max(0, n_rows), 0), dtype=np.float32)
            if symbol_col is not None:
                symbols = self._as_1d(frame[symbol_col]).astype(str, copy=False)[: max(0, n_rows)]
        elif isinstance(frame, dict):
            if "X" in frame:
                raw_X = frame.get("X")
                names = frame.get("feature_names")
                if names is not None:
                    input_names = [str(c) for c in list(names)]
                if "symbol" in frame:
                    symbols = np.asarray(frame.get("symbol"), dtype=str).reshape(-1)
            else:
                names = [str(k) for k in frame.keys() if str(k) != "symbol"]
                if names:
                    cols = [self._as_numeric_1d(frame.get(name), dtype=np.float32).reshape(-1) for name in names]
                    n_rows = min(int(c.size) for c in cols) if cols else 0
                    raw_X = np.column_stack([c[:n_rows] for c in cols]) if n_rows > 0 else np.zeros((0, 0), dtype=np.float32)
                    input_names = names
                else:
                    raw_X = np.zeros((0, 0), dtype=np.float32)
                if "symbol" in frame:
                    symbols = self._as_1d(frame.get("symbol")).astype(str, copy=False)

        X_arr = self._as_2d_float32(raw_X)
        if input_names:
            return self._map_named_features(X_arr, input_names, symbols=symbols)

        if X_arr.shape[1] == len(self.feature_columns):
            return X_arr
        n = int(X_arr.shape[0])
        out = np.zeros((n, int(len(self.feature_columns))), dtype=np.float32)
        cols = min(int(X_arr.shape[1]), int(len(self.feature_columns)))
        if cols > 0:
            out[:, :cols] = X_arr[:, :cols]
        return out

    def predict_proba(self, frame: Any) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("MetaBlender has not been fitted")
        if self.constant_proba is not None:
            n = len(frame)
            return np.tile(self.constant_proba.reshape(1, -1), (n, 1))

        X_np = self._coerce_predict_input(frame)
        classes = self.proba_classes
        if classes is None:
            try:
                classes = [int(c) for c in list(self.model.classes_)]
            except Exception:
                classes = None
        raw = self.model.predict_proba(X_np)
        return pad_probs(raw, classes=classes)

    def save(self, path: Path) -> None:
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "proba_classes": self.proba_classes,
            "constant_proba": self.constant_proba,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> MetaBlender:
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        cols = payload.get("feature_columns")
        if cols is not None:
            instance.feature_columns = [str(c) for c in list(cols)]
        else:
            instance.feature_columns = None
        instance.proba_classes = payload.get("proba_classes")
        instance.constant_proba = payload.get("constant_proba")
        return instance

