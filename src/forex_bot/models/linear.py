from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .base import ExpertModel
from .label_utils import margins_to_probs, probs_to_three_class, remap_labels_neutral_buy_sell

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.tree import DecisionTreeClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    SGDClassifier = None  # type: ignore
    DecisionTreeClassifier = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    from vowpalwabbit.sklearn_vw import VWClassifier

    VW_AVAILABLE = True
except Exception:
    VWClassifier = None  # type: ignore
    VW_AVAILABLE = False

try:
    from river.tree import HoeffdingTreeClassifier as RiverHoeffdingTreeClassifier

    RIVER_AVAILABLE = True
except Exception:
    RiverHoeffdingTreeClassifier = None  # type: ignore
    RIVER_AVAILABLE = False


def _canon_y(y: Any | np.ndarray) -> np.ndarray:
    # Canonical class order: 0=neutral, 1=buy, 2=sell.
    return remap_labels_neutral_buy_sell(y)


def _as_numeric_matrix(x: Any, cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if x is None:
        return np.zeros((0, 0), dtype=np.float32), list(cols or [])
    if hasattr(x, "select_dtypes"):
        x_num = x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = np.asarray(x_num.to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        source_names = [str(c) for c in x_num.columns]
        if cols:
            target_names = [str(c) for c in cols]
            out = np.zeros((arr.shape[0], len(target_names)), dtype=np.float32)
            name_to_idx = {name: i for i, name in enumerate(source_names)}
            for j, name in enumerate(target_names):
                idx = name_to_idx.get(name)
                if idx is not None:
                    out[:, j] = arr[:, idx]
            return out, target_names
        return arr, source_names
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    n_cols = int(arr.shape[1]) if arr.ndim == 2 else 0
    if cols and len(cols) == n_cols:
        names = [str(c) for c in cols]
    elif cols and len(cols) > 0:
        names = [str(c) for c in cols]
        target = len(names)
        out = np.zeros((arr.shape[0], target), dtype=np.float32)
        keep = min(target, n_cols)
        if keep > 0:
            out[:, :keep] = arr[:, :keep]
        arr = out
    else:
        names = [f"f{i}" for i in range(n_cols)]
    return arr, names


def _pad_probs_with_classes(probs: np.ndarray, classes: np.ndarray | list[int] | None) -> np.ndarray:
    return probs_to_three_class(probs, classes)


def _decision_to_probs(decision: np.ndarray) -> np.ndarray:
    return margins_to_probs(decision)


def _iter_feature_dict_rows(x: Any):
    """
    Convert numeric rows to feature dicts with low overhead.
    """
    arr, names = _as_numeric_matrix(x)
    n_feat = int(arr.shape[1]) if arr.ndim == 2 else 0
    for row in arr:
        yield {names[j]: float(row[j]) for j in range(n_feat)}


class _LinearBase(ExpertModel):
    model_name: str = "linear_base"

    def __init__(self, **kwargs: Any) -> None:
        self.model: Any = None
        self.feature_columns: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self.constant_proba: np.ndarray | None = None
        self.params = dict(kwargs)

    def _build_model(self) -> Any:
        raise NotImplementedError

    def _prepare_x(self, x: Any, cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        return _as_numeric_matrix(x, cols)

    def fit(self, x: Any, y: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if not SKLEARN_AVAILABLE:
            logger.warning("%s skipped: scikit-learn not available.", self.model_name)
            self.model = None
            return
        if x is None or len(x) == 0:
            self.model = None
            return
        x_arr, x_cols = self._prepare_x(x)
        self.feature_columns = list(x_cols)
        y_arr = _canon_y(y)
        uniq, cnt = np.unique(y_arr, return_counts=True)
        if len(uniq) < 2:
            pri = np.zeros(3, dtype=float)
            c = int(uniq[0]) if len(uniq) else 0
            pri[c] = 1.0
            self.constant_proba = pri
            self.classes_ = np.array([0, 1, 2], dtype=int)
            self.model = None
            return
        self.constant_proba = None
        self.model = self._build_model()
        self.model.fit(x_arr, y_arr)
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.array([0, 1, 2], dtype=int)))

    def predict_proba(self, x: Any, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        if x is None:
            return np.zeros((0, 3), dtype=float)
        x_arr, _x_cols = self._prepare_x(x, self.feature_columns)
        n_rows = int(x_arr.shape[0])
        if self.constant_proba is not None:
            return np.tile(self.constant_proba.reshape(1, -1), (n_rows, 1))
        if self.model is None:
            return np.zeros((n_rows, 3), dtype=float)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_arr)
            classes = getattr(self.model, "classes_", self.classes_)
            return _pad_probs_with_classes(probs, classes)
        decision = self.model.decision_function(x_arr)
        probs = _decision_to_probs(decision)
        return _pad_probs_with_classes(probs, self.classes_)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "classes_": self.classes_,
            "constant_proba": self.constant_proba,
            "params": self.params,
        }
        joblib.dump(payload, p / f"{self.model_name}.joblib")

    def load(self, path: str) -> None:
        fp = Path(path) / f"{self.model_name}.joblib"
        if not fp.exists():
            return
        try:
            payload = joblib.load(fp)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.feature_columns = payload.get("feature_columns")
                self.classes_ = payload.get("classes_")
                self.constant_proba = payload.get("constant_proba")
                self.params = dict(payload.get("params") or self.params)
        except Exception as exc:
            logger.warning("Failed loading %s: %s", self.model_name, exc)


class ElasticNetExpert(_LinearBase):
    model_name = "elasticnet"

    def _build_model(self) -> Any:
        if SGDClassifier is None:
            raise RuntimeError("scikit-learn missing")
        alpha = float(self.params.get("alpha", 1e-4) or 1e-4)
        l1_ratio = float(self.params.get("l1_ratio", 0.5) or 0.5)
        max_iter = int(self.params.get("max_iter", 2000) or 2000)
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )


class BayesianLogitExpert(_LinearBase):
    """
    Robust Bayesian-style logistic baseline.
    Uses stronger regularization as a practical approximation to informative priors.
    """

    model_name = "bayes_logit"

    def _build_model(self) -> Any:
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn missing")
        c_val = float(self.params.get("C", 0.5) or 0.5)
        return LogisticRegression(
            max_iter=int(self.params.get("max_iter", 800) or 800),
            solver="lbfgs",
            class_weight="balanced",
            C=max(1e-4, c_val),
            random_state=42,
        )


class OnlinePassiveAggressiveExpert(_LinearBase):
    model_name = "online_pa"

    def _build_model(self) -> Any:
        if SGDClassifier is None:
            raise RuntimeError("scikit-learn missing")
        # sklearn deprecated PassiveAggressiveClassifier in favor of SGDClassifier(pa1).
        return SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=max(1e-6, float(self.params.get("C", 0.5) or 0.5)),
            max_iter=int(self.params.get("max_iter", 2000) or 2000),
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )


class OnlineHoeffdingExpert(_LinearBase):
    model_name = "online_hoeffding"

    def _build_model(self) -> Any:
        # River Hoeffding is optional; use a stable shallow tree fallback when unavailable.
        if RIVER_AVAILABLE:
            return RiverHoeffdingTreeClassifier()
        if DecisionTreeClassifier is None:
            raise RuntimeError("scikit-learn missing")
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)

    def fit(self, x: Any, y: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if RIVER_AVAILABLE:
            x_arr, x_cols = self._prepare_x(x)
            self.feature_columns = list(x_cols)
            y_arr = _canon_y(y)
            model = self._build_model()
            for feats, yi in zip(_iter_feature_dict_rows(x_arr), y_arr, strict=False):
                model.learn_one(feats, int(yi))
            self.model = model
            self.classes_ = np.array([0, 1, 2], dtype=int)
            self.constant_proba = None
            return
        super().fit(x, y, **kwargs)

    def predict_proba(self, x: Any, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        if RIVER_AVAILABLE and self.model is not None:
            if x is None or len(x) == 0:
                return np.zeros((0, 3), dtype=float)
            x_arr, _x_cols = self._prepare_x(x, self.feature_columns)
            out = np.zeros((len(x_arr), 3), dtype=float)
            for i, row in enumerate(_iter_feature_dict_rows(x_arr)):
                p = self.model.predict_proba_one(row) or {}
                out[i, 0] = float(p.get(0, 0.0))
                out[i, 1] = float(p.get(1, 0.0))
                out[i, 2] = float(p.get(2, 0.0))
                s = float(out[i].sum())
                if s <= 0:
                    out[i, 0] = 1.0
                else:
                    out[i] /= s
            return out
        return super().predict_proba(x, **kwargs)


class VowpalWabbitExpert(_LinearBase):
    model_name = "vw"

    def _build_model(self) -> Any:
        if not VW_AVAILABLE:
            raise RuntimeError("vowpalwabbit not available")
        return VWClassifier(loss_function="logistic", oaa=3, passes=3, random_seed=42)

    def fit(self, x: Any, y: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if not VW_AVAILABLE:
            logger.warning("vw skipped: vowpalwabbit not available.")
            self.model = None
            return
        super().fit(x, y, **kwargs)


