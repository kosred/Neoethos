from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..core.config import Settings
from ..domain.events import PreparedDataset, SignalResult
from ..training.calibration import ProbabilityCalibrator
from ..training.conformal import ConformalClassifierGate
from ..training.evaluation import pad_probs, probs_to_signals
from ..training.persistence_service import PersistenceService

logger = logging.getLogger(__name__)
try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore


def _is_dataframe_like(value: Any) -> bool:
    return bool(
        hasattr(value, "columns")
        and hasattr(value, "index")
        and callable(getattr(value, "to_numpy", None))
    )


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "__getitem__"))


def _frame_columns(value: Any) -> list[str]:
    cols = getattr(value, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_index(value: Any) -> Any | None:
    return getattr(value, "index", None)


def _frame_extract_column(value: Any, name: str) -> np.ndarray | None:
    col_name = str(name).strip().lower()
    if isinstance(value, dict):
        for key, arr in value.items():
            if str(key).strip().lower() == col_name:
                return np.asarray(arr).reshape(-1)
        return None
    if not _is_frame_like(value):
        return None
    resolved = None
    for col in _frame_columns(value):
        if str(col).strip().lower() == col_name:
            resolved = col
            break
    if resolved is None:
        return None
    try:
        raw = value[resolved]  # type: ignore[index]
        arr = raw.to_numpy(copy=False) if hasattr(raw, "to_numpy") else np.asarray(raw)
        return np.asarray(arr).reshape(-1)
    except Exception:
        return None


def _vector_scalar_at(values: np.ndarray | None, idx: int, *, default: float = 0.0) -> float:
    if values is None:
        return float(default)
    try:
        arr = np.asarray(values).reshape(-1)
        if arr.size <= 0:
            return float(default)
        pos = int(idx)
        if pos < 0:
            pos = int(arr.size + pos)
        if pos < 0 or pos >= int(arr.size):
            return float(default)
        with np.errstate(all="ignore"):
            out = float(arr[pos])
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _column_index_mapping(src_names: list[str], dst_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    dst_lookup = {str(name): i for i, name in enumerate(dst_names)}
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for src_i, raw_name in enumerate(src_names):
        dst_i = dst_lookup.get(str(raw_name))
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

    if _fb is not None and hasattr(_fb, "align_feature_matrix"):
        try:
            out = _fb.align_feature_matrix(src, src_idx, dst_idx, width)
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2 and arr.shape == (rows, width):
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        except Exception:
            pass

    out = np.zeros((rows, width), dtype=np.float32)
    out[:, dst_idx] = src[:, src_idx]
    return out


class _NumpyFrame:
    def __init__(self, data: dict[str, Any], index: Any, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)
        self.attrs = dict(attrs or {})

    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def to_numpy(self, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if not self.columns:
            arr = np.zeros((len(self), 0), dtype=np.float32)
        else:
            arr = np.column_stack([np.asarray(self._data[str(col)]).reshape(-1) for col in self.columns])
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        elif copy:
            arr = np.array(arr, copy=True)
        return arr


def _frame_from_matrix(matrix: Any, columns: list[str], index: Any, *, template: Any | None = None) -> Any:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    idx = np.asarray(index).reshape(-1) if index is not None else np.arange(int(arr.shape[0]), dtype=np.int64)
    if idx.size != int(arr.shape[0]):
        idx = np.arange(int(arr.shape[0]), dtype=np.int64)
    data = {str(col): arr[:, i] if i < arr.shape[1] else np.zeros(arr.shape[0], dtype=np.float32) for i, col in enumerate(columns)}
    attrs = getattr(template, "attrs", None)
    attrs_dict = dict(attrs) if isinstance(attrs, dict) else None
    if template is not None:
        frame_cls = template.__class__
        try:
            return frame_cls(data, index=idx, columns=columns)
        except Exception:
            try:
                return frame_cls(arr, index=idx, columns=columns)
            except Exception:
                pass
    return _NumpyFrame(data, idx, attrs=attrs_dict)


@dataclass(slots=True)
class _ModelProba:
    name: str
    proba: np.ndarray


class SignalEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        self.logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))
        self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        self.models: dict[str, Any] = {}
        self.meta_blender: Any | None = None
        self._onnx = None
        self.calibrators: dict[str, ProbabilityCalibrator] = {}
        self.conformal_gate: ConformalClassifierGate | None = None
        self.selected_features: list[str] = []
        self.selected_features_by_regime: dict[str, list[str]] = {}

    def _load_selected_features(self) -> list[str]:
        path = self.models_dir / "selected_features.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(c) for c in payload if str(c).strip()]
        except Exception as exc:
            logger.debug("SignalEngine: failed to load selected features: %s", exc)
        return []

    def _load_selected_features_by_regime(self) -> dict[str, list[str]]:
        path = self.models_dir / "selected_features_by_regime.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            out: dict[str, list[str]] = {}
            for key, vals in payload.items():
                bucket = str(key).strip().lower()
                if not bucket or not isinstance(vals, list):
                    continue
                cols = [str(c) for c in vals if str(c).strip()]
                if cols:
                    out[bucket] = cols
            return out
        except Exception as exc:
            logger.debug("SignalEngine: failed to load per-regime selected features: %s", exc)
            return {}

    def _load_calibrators(self) -> dict[str, ProbabilityCalibrator]:
        path = self.models_dir / "calibrators.joblib"
        if not path.exists():
            return {}
        try:
            payload = joblib.load(path)
            if isinstance(payload, dict):
                out = {str(k): v for k, v in payload.items() if isinstance(v, ProbabilityCalibrator)}
                return out
        except Exception as exc:
            logger.debug("SignalEngine: failed to load calibrators: %s", exc)
        return {}

    def _load_conformal_gate(self) -> ConformalClassifierGate | None:
        path = self.models_dir / "conformal_gate.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if not bool(payload.get("enabled", True)):
                return None
            gate = ConformalClassifierGate(alpha=float(payload.get("alpha", 0.10) or 0.10))
            gate.qhat = float(payload.get("qhat", 1.0) or 1.0)
            gate.fitted = bool(payload.get("fitted", False))
            gate.n_calib = int(payload.get("n_calib", 0) or 0)
            return gate if gate.fitted else None
        except Exception as exc:
            logger.debug("SignalEngine: failed to load conformal gate: %s", exc)
            return None

    def load_models(self, models_dir: str | Path | None = None) -> None:
        if models_dir is not None:
            self.models_dir = Path(models_dir)
            self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        try:
            self.models, self.meta_blender = self.persistence.load_models()
        except Exception as exc:
            logger.warning("Failed to load models: %s", exc)
            self.models = {}
            self.meta_blender = None
        self.selected_features = self._load_selected_features()
        self.selected_features_by_regime = self._load_selected_features_by_regime()
        self.calibrators = self._load_calibrators()
        self.conformal_gate = self._load_conformal_gate()
        # Optional ONNX inference path
        use_onnx = str(os.environ.get("FOREX_BOT_USE_ONNX", "")).strip().lower() in {"1", "true", "yes", "on"}
        if use_onnx:
            try:
                from ..models.onnx_exporter import ONNXExporter

                exporter = ONNXExporter(str(self.models_dir))
                exporter.load_models()
                self._onnx = exporter
            except Exception as exc:
                logger.warning("ONNX load failed: %s", exc)
                self._onnx = None

    @staticmethod
    def _as_2d_float32(values: Any) -> np.ndarray:
        arr = None
        if hasattr(values, "to_numpy"):
            try:
                arr = np.asarray(values.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
            except Exception:
                arr = None
        if arr is None and _is_frame_like(values):
            cols = _frame_columns(values)
            mats: list[np.ndarray] = []
            n_rows = 0
            for col in cols:
                vec = _frame_extract_column(values, col)
                if vec is None:
                    continue
                try:
                    arr_col = np.asarray(vec, dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                mats.append(np.nan_to_num(arr_col, nan=0.0, posinf=0.0, neginf=0.0))
                n_rows = max(n_rows, int(arr_col.size))
            if mats:
                arr = np.zeros((n_rows, len(mats)), dtype=np.float32)
                for j, vec in enumerate(mats):
                    take = min(n_rows, int(vec.size))
                    if take > 0:
                        arr[:take, j] = vec[:take]
        if arr is None:
            arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not arr.flags.writeable:
            arr = arr.copy()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    @staticmethod
    def _resolve_feature_names(feature_names: Any, n_features: int) -> list[str]:
        try:
            names = [str(c) for c in list(feature_names)]
        except Exception:
            names = []
        if len(names) != int(n_features):
            return [f"f{i}" for i in range(int(n_features))]
        return names

    @staticmethod
    def _resolve_index(index_like: Any, n_rows: int) -> np.ndarray:
        if index_like is None:
            return np.arange(int(n_rows), dtype=np.int64)
        try:
            arr = np.asarray(index_like).reshape(-1)
        except Exception:
            return np.arange(int(n_rows), dtype=np.int64)
        if arr.size != int(n_rows):
            return np.arange(int(n_rows), dtype=np.int64)
        return arr

    def _align_selected_features(
        self,
        X: Any,
        regime_bucket: str | None = None,
        *,
        feature_names: list[str] | None = None,
    ) -> tuple[Any, list[str] | None]:
        cols: list[str] = []
        use_regime = bool(getattr(self.settings.models, "l1_feature_selection_per_regime", True))
        if use_regime and regime_bucket:
            cols = list(self.selected_features_by_regime.get(str(regime_bucket).strip().lower(), []) or [])
        if not cols:
            cols = list(self.selected_features or [])
        if not cols:
            return X, list(feature_names) if feature_names is not None else None
        if _is_dataframe_like(X):
            X_arr = self._as_2d_float32(X)
            names = _frame_columns(X)
            if not names:
                names = self._resolve_feature_names(feature_names, X_arr.shape[1])
            src_cols, dst_cols = _column_index_mapping(names, cols)
            aligned = _align_feature_matrix(X_arr, src_cols, dst_cols, dst_width=len(cols))
            return _frame_from_matrix(aligned, cols, _frame_index(X), template=X), list(cols)

        if _is_frame_like(X):
            frame_names = _frame_columns(X)
            if frame_names:
                feature_names = frame_names
        X_arr = self._as_2d_float32(X)
        names = self._resolve_feature_names(feature_names, X_arr.shape[1])
        src_cols, dst_cols = _column_index_mapping(names, cols)
        aligned = _align_feature_matrix(X_arr, src_cols, dst_cols, dst_width=len(cols))
        if _is_frame_like(X):
            return _frame_from_matrix(aligned, cols, _frame_index(X), template=X), list(cols)
        return aligned, list(cols)

    def _predict_model_proba(self, name: str, model: Any, X: Any) -> np.ndarray | None:
        try:
            proba = pad_probs(model.predict_proba(X))
            if proba.shape[0] != len(X):
                return None
            cal = self.calibrators.get(name)
            if cal is not None:
                proba = cal.predict_proba(proba)
            return proba
        except Exception as exc:
            logger.debug("Model prediction failed (%s): %s", name, exc)
            return None

    def _collect_model_probas(self, X: Any, models_subset: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        models = models_subset if models_subset is not None else self.models
        out: dict[str, np.ndarray] = {}
        for name, model in models.items():
            proba = self._predict_model_proba(name, model, X)
            if proba is not None:
                out[name] = proba
        return out

    def _ensemble_proba(
        self,
        X: Any,
        models_subset: dict[str, Any] | None = None,
        *,
        proba_cache: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        cache = proba_cache if proba_cache is not None else self._collect_model_probas(X, models_subset=models_subset)
        probas = list(cache.values())
        if not probas:
            return np.zeros((len(X), 3), dtype=float)
        return np.mean(probas, axis=0)

    def _meta_features(
        self,
        X: Any,
        models_subset: dict[str, Any] | None = None,
        *,
        proba_cache: dict[str, np.ndarray] | None = None,
    ) -> Any:
        cache = proba_cache if proba_cache is not None else self._collect_model_probas(X, models_subset=models_subset)
        names: list[str] = []
        cols: list[np.ndarray] = []
        for name, proba in cache.items():
            names.append(f"{name}_buy")
            cols.append(np.asarray(proba[:, 1], dtype=np.float32).reshape(-1, 1))
        if not cols:
            return {"X": np.zeros((len(X), 1), dtype=np.float32), "feature_names": ["dummy"]}
        return {"X": np.column_stack(cols).astype(np.float32, copy=False), "feature_names": names}

    def _infer_regime_bucket(self, X: Any, last_idx: int, *, feature_names: list[str] | None = None) -> str:
        # Heuristic regime routing from ADX + realized volatility proxy.
        adx = 0.0
        vol = self._latest_market_volatility(X, last_idx, feature_names=feature_names)
        if _is_frame_like(X):
            adx_vec = _frame_extract_column(X, "adx")
            if adx_vec is None:
                for c in _frame_columns(X):
                    if str(c).strip().lower().endswith("_adx"):
                        adx_vec = _frame_extract_column(X, c)
                        if adx_vec is not None:
                            break
            adx = _vector_scalar_at(adx_vec, last_idx, default=0.0)
        else:
            arr = self._as_2d_float32(X)
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                if _is_frame_like(X):
                    names = self._resolve_feature_names(_frame_columns(X), arr.shape[1])
                else:
                    names = self._resolve_feature_names(feature_names, arr.shape[1])
                idx_map = {str(name): i for i, name in enumerate(names)}
                adx_idx = idx_map.get("adx")
                if adx_idx is None:
                    for name, pos in idx_map.items():
                        if name.endswith("_adx"):
                            adx_idx = pos
                            break
                if adx_idx is not None and adx_idx < arr.shape[1]:
                    with np.errstate(all="ignore"):
                        adx = float(arr[last_idx, int(adx_idx)])
        trend_thr = float(getattr(self.settings.risk, "regime_adx_trend", 25.0) or 25.0)
        range_thr = float(getattr(self.settings.risk, "regime_adx_range", 20.0) or 20.0)
        if adx >= trend_thr:
            return "trend"
        if adx <= range_thr:
            return "range"
        v_ref = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
        if v_ref > 0 and vol > (v_ref * 1.5):
            return "high_vol"
        return "neutral"

    def _select_models_for_regime(
        self,
        X: Any,
        last_idx: int,
        *,
        feature_names: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        if not bool(getattr(self.settings.models, "regime_router_enabled", True)):
            return self.models, "neutral"
        if not self.models:
            return {}, "neutral"
        bucket = self._infer_regime_bucket(X, last_idx, feature_names=feature_names)
        if bucket == "trend":
            candidates = set(getattr(self.settings.models, "regime_trend_models", []) or [])
        elif bucket == "range":
            candidates = set(getattr(self.settings.models, "regime_range_models", []) or [])
        else:
            candidates = set(getattr(self.settings.models, "regime_neutral_models", []) or [])
        chosen = {k: v for k, v in self.models.items() if (not candidates or k in candidates)}
        min_models = int(getattr(self.settings.models, "regime_router_min_models", 2) or 2)
        if len(chosen) < max(1, min_models):
            return self.models, bucket
        return chosen, bucket

    def _predict_proba(
        self,
        X: Any,
        models_subset: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        model_probas = self._collect_model_probas(X, models_subset=models_subset)
        use_all_models = models_subset is None or len(models_subset) == len(self.models)
        if self._onnx is not None and use_all_models:
            try:
                return self._onnx.predict_with_meta_blender(X), model_probas
            except Exception:
                pass
        if self.meta_blender is not None:
            try:
                feats = self._meta_features(X, models_subset=models_subset, proba_cache=model_probas)
                return pad_probs(self.meta_blender.predict_proba(feats)), model_probas
            except Exception:
                pass
        return self._ensemble_proba(X, models_subset=models_subset, proba_cache=model_probas), model_probas

    @staticmethod
    def _latest_market_volatility(X: Any, last_idx: int, *, feature_names: list[str] | None = None) -> float:
        if X is None or len(X) == 0:
            return 0.0
        close = 0.0
        atr_val = 0.0
        if _is_frame_like(X):
            close_vec = None
            for c in ("close", "price"):
                close_vec = _frame_extract_column(X, c)
                if close_vec is not None:
                    break
            atr_vec = None
            for c in ("atr14", "atr", "M1_atr14", "M5_atr14"):
                atr_vec = _frame_extract_column(X, c)
                if atr_vec is not None:
                    break
            close = _vector_scalar_at(close_vec, last_idx, default=0.0)
            atr_val = _vector_scalar_at(atr_vec, last_idx, default=0.0)
        else:
            arr = SignalEngine._as_2d_float32(X)
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                return 0.0
            names = SignalEngine._resolve_feature_names(feature_names, arr.shape[1])
            idx_map = {str(name): i for i, name in enumerate(names)}
            close_idx = None
            for c in ("close", "Close", "price"):
                if c in idx_map:
                    close_idx = idx_map[c]
                    break
            atr_idx = None
            for c in ("atr14", "atr", "ATR", "M1_atr14", "M5_atr14"):
                if c in idx_map:
                    atr_idx = idx_map[c]
                    break
            if close_idx is not None:
                close = float(arr[last_idx, int(close_idx)])
            if atr_idx is not None:
                atr_val = float(arr[last_idx, int(atr_idx)])
        if close > 0 and atr_val > 0:
            return float(max(0.0, atr_val / close))
        return 0.0

    @staticmethod
    def _ensemble_disagreement(last_model_probs: list[np.ndarray]) -> float:
        if not last_model_probs:
            return 0.0
        mat = np.asarray(last_model_probs, dtype=float)
        if mat.ndim != 2 or mat.shape[0] <= 1:
            return 0.0
        preds = np.argmax(mat, axis=1)
        vals, cnts = np.unique(preds, return_counts=True)
        if len(vals) == 0:
            return 0.0
        majority = int(np.max(cnts))
        disagree = 1.0 - (float(majority) / float(max(1, len(preds))))
        return float(max(0.0, min(1.0, disagree)))

    def generate_ensemble_signals(self, dataset: PreparedDataset) -> SignalResult:
        X = dataset.X
        if X is None or len(X) == 0:
            return SignalResult(
                signal=0,
                confidence=0.0,
                model_votes={},
                regime="Normal",
                meta_features={},
                probs=np.zeros(3, dtype=float),
                signals=np.zeros(0, dtype=int),
            )

        if _is_dataframe_like(X):
            X_raw = X
            feature_names = [str(c) for c in list(X_raw.columns)]
        elif _is_frame_like(X):
            X_raw = self._as_2d_float32(X)
            feature_names = self._resolve_feature_names(_frame_columns(X), X_raw.shape[1])
        else:
            X_raw = self._as_2d_float32(X)
            feature_names = self._resolve_feature_names(getattr(dataset, "feature_names", None), X_raw.shape[1])

        last_idx = -1 if len(X_raw) > 0 else None
        active_models, regime_bucket = self._select_models_for_regime(
            X_raw,
            last_idx or -1,
            feature_names=feature_names,
        )
        X, _ = self._align_selected_features(X_raw, regime_bucket=regime_bucket, feature_names=feature_names)

        probas, model_proba_cache = self._predict_proba(X, models_subset=active_models)
        signals = probs_to_signals(probas)
        last_idx = -1 if len(signals) > 0 else None
        if last_idx is None:
            last_signal = 0
            confidence = 0.0
            last_probs = np.zeros(3, dtype=float)
            market_volatility = 0.0
            disagreement = 0.0
            uncertainty_value = 0.0
            conformal_set_size = 1
            conformal_abstained = False
        else:
            last_signal = int(signals[last_idx])
            last_probs = probas[last_idx]
            confidence = float(np.max(last_probs)) if last_probs is not None else 0.0
            market_volatility = self._latest_market_volatility(X_raw, last_idx, feature_names=feature_names)
            disagreement = 0.0
            uncertainty_value = max(0.0, min(1.0, 1.0 - confidence))
            conformal_set_size = 1
            conformal_abstained = False

            if self.conformal_gate is not None and bool(getattr(self.settings.risk, "conformal_enabled", True)):
                min_size = int(getattr(self.settings.risk, "conformal_abstain_min_set_size", 3) or 3)
                abstain, set_size = self.conformal_gate.should_abstain(last_probs, min_set_size=min_size)
                conformal_set_size = int(set_size)
                conformal_abstained = bool(abstain)
                if abstain:
                    last_signal = 0

        votes: dict[str, float] = {}
        model_last_probs: list[np.ndarray] = []
        for name, proba in model_proba_cache.items():
            if last_idx is None:
                continue
            votes[name] = float(proba[last_idx, 1] - proba[last_idx, 2])
            model_last_probs.append(np.asarray(proba[last_idx], dtype=float))

        if last_idx is not None and model_last_probs:
            disagreement = self._ensemble_disagreement(model_last_probs)
            uncertainty_value = float(max(0.0, min(1.0, max(1.0 - confidence, disagreement))))

        uncertainty_series = None
        if last_idx is not None:
            uncertainty_series = np.asarray([float(uncertainty_value)], dtype=np.float64)

        signals_out = np.asarray(signals, dtype=int)

        return SignalResult(
            signal=last_signal,
            confidence=confidence,
            model_votes=votes,
            regime=str(regime_bucket or "neutral"),
            meta_features={
                "ensemble_disagreement": float(disagreement),
                "market_volatility": float(market_volatility),
                "uncertainty": float(uncertainty_value),
                "regime_bucket": str(regime_bucket or "neutral"),
                "conformal_set_size": int(conformal_set_size),
                "conformal_abstained": bool(conformal_abstained),
            },
            probs=np.asarray(last_probs, dtype=float),
            signals=signals_out,
            uncertainty=uncertainty_series,
        )


__all__ = ["SignalEngine"]

